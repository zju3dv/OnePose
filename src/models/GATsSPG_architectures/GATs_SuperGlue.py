from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GATs import GraphAttentionLayer


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


def buildAdjMatrix(num_2d, num_3d):
    num_leaf = int(num_2d / num_3d)

    adj_matrix = torch.zeros(num_3d, num_2d)
    for i in range(num_3d):
        adj_matrix[i, num_leaf*i: num_leaf*(i+1)] = 1 / num_leaf
    return adj_matrix.cuda()


class AttentionalGNN(nn.Module):
    
    def __init__(
        self, 
        feature_dim: int, 
        layer_names: list,
        include_self: bool,
        additional: bool,
        with_linear_transform: bool
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=256, 
                out_features=256, 
                dropout=0.6, 
                alpha=0.2, 
                concat=True,
                include_self=include_self,
                additional=additional,
                with_linear_transform=with_linear_transform, 
            ) if i % 3 == 0 else AttentionPropagation(feature_dim, 4)
            for i in range(len(layer_names))
        ])
        self.names = layer_names
    
    def forward(self, desc2d_query, desc3d_db, desc2d_db):
        for layer, name in zip(self.layers, self.names):
            if name == 'GATs':
                desc2d_db_ = torch.einsum('bdn->bnd', desc2d_db)
                desc3d_db_ = torch.einsum('bdn->bnd', desc3d_db)
                desc3d_db_ = layer(desc2d_db_, desc3d_db_)
                desc3d_db = torch.einsum('bnd->bdn', desc3d_db_)
            elif name == 'cross':
                layer.attn.prob = []
                src0, src1 = desc3d_db, desc2d_query # [b, c, l1], [b, c, l2]
                delta0, delta1 = layer(desc2d_query, src0), layer(desc3d_db, src1)
                desc2d_query, desc3d_db = (desc2d_query + delta0), (desc3d_db + delta1)
            elif name == 'self':
                layer.attn.prob = []
                src0, src1 = desc2d_query, desc3d_db
                delta0, delta1 = layer(desc2d_query, src0), layer(desc3d_db, src1)
                desc2d_query, desc3d_db = (desc2d_query + delta0), (desc3d_db + delta1)
        
        return desc2d_query, desc3d_db
    

def linear_attention(query, key, value):
    eps = 1e-6
    query = F.elu(query) + 1
    key = F.elu(key) + 1

    v_length = value.size(3)
    value = value / v_length

    KV = torch.einsum('bdhm,bqhm->bqdh', key, value)
    Z = 1 / (torch.einsum('bdhm,bdh->bhm', query, key.sum(3)) + eps)
    queried_values = torch.einsum('bdhm,bqdh,bhm->bqhm', query, KV, Z) * v_length
    return queried_values.contiguous()


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivity"""
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
    
    def forward(self, query, key, value):
        batch_dim = query.size(0)

        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x = linear_attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.)
    
    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) # [b, 2c, 1000] / [b, 2c, 2000]


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < n -1:
            if do_bn: 
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """
    def __init__(self, inp_dim, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([inp_dim] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class GATsSuperGlue(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.match_type = hparams['match_type']

        self.kenc_2d = KeypointEncoder(
                        inp_dim=3,
                        feature_dim=hparams['descriptor_dim'],
                        layers=hparams['keypoints_encoder']
                    )

        self.kenc_3d = KeypointEncoder(
                        inp_dim=4,
                        feature_dim=hparams['descriptor_dim'],
                        layers=hparams['keypoints_encoder']
                    )
        
        GNN_layers = ['GATs', 'self', 'cross'] * 4
        self.gnn = AttentionalGNN(
                        feature_dim=hparams['descriptor_dim'],
                        layer_names=GNN_layers,
                        include_self=hparams['include_self'],
                        additional=hparams['additional'],
                        with_linear_transform=hparams['with_linear_transform']
                    )
        self.final_proj = nn.Conv1d(
                            in_channels=hparams['descriptor_dim'],
                            out_channels=hparams['descriptor_dim'],
                            kernel_size=1,
                            bias=True
                        )
        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
    
    def forward(self, data):
        """ 
        Keys of data:
            keypoints2d: [b, n1, 2]
            keypoints3d: [b, n2, 3]
            descriptors2d_query: [b, dim, n1]
            descriptors3d_db: [b, dim, n2]
            descriptors2d_db: [b, dim, n2 * num_leaf]
            scores2d_query: [b, n1, 1]
            scores3d_db: [b, n2, 1]
            scores2d_db: [b, n2 * num_leaf, 1]
        """
        kpts2d, kpts3d = data['keypoints2d'].float(), data['keypoints3d'].float()
        desc2d_query = data['descriptors2d_query'].float()
        desc3d_db, desc2d_db = data['descriptors3d_db'].float(), data['descriptors2d_db'].float()

        if kpts2d.shape[1] == 0 or kpts3d.shape[1] == 0:
            shape0, shape1 = kpts2d.shape[:-1], kpts3d.shape[:-1]
            return {
                'matches0': kpts2d.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts3d.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts2d.new_zeros(shape0)[0],
                'matching_scores1': kpts3d.new_zeros(shape1)[0],
                'skip_train': True
            }

        # Multi-layer Transformer network
        desc2d_query, desc3d_db = self.gnn(desc2d_query, desc3d_db, desc2d_db)

        # Final MLP projection
        mdesc2d_query, mdesc3d_db = self.final_proj(desc2d_query), self.final_proj(desc3d_db)

        # Normalize mdesc to avoid NaN
        mdesc2d_query = F.normalize(mdesc2d_query, p=2, dim=1)
        mdesc3d_db = F.normalize(mdesc3d_db, p=2, dim=1)

        # Get the matches with score above "match_threshold"
        if self.match_type == "softmax":
            scores = torch.einsum('bdn,bdm->bnm', mdesc2d_query, mdesc3d_db) / self.hparams['scale_factor']
            conf_matrix = F.softmax(scores, 1) * F.softmax(scores, 2)

            max0, max1 = conf_matrix[:, :, :].max(2), conf_matrix[:, :, :].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = conf_matrix.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values, zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.hparams['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            pred = {
                'matches0': indices0[0], # use -1 for invalid match
                'matches1': indices1[0], # use -1 for invalid match
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0],
            }
        else:
            raise NotImplementedError
        
        return pred, conf_matrix
