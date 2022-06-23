import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        dropout, 
        alpha, 
        concat=True, 
        include_self=True,
        additional=False,
        with_linear_transform=True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.include_self = include_self
        self.with_linear_transform = with_linear_transform
        self.additional = additional
    
    def forward(self, h_2d, h_3d):
        b, n1, dim = h_3d.shape
        b, n2, dim = h_2d.shape
        num_leaf = int(n2 / n1)

        wh_2d = torch.matmul(h_2d, self.W)
        wh_3d = torch.matmul(h_3d, self.W)

        e = self._prepare_attentional_mechanism_input(wh_2d, wh_3d, num_leaf, self.include_self)
        attention = F.softmax(e, dim=2)

        h_2d = torch.reshape(h_2d, (b, n1, num_leaf, dim))
        wh_2d = torch.reshape(wh_2d, (b, n1, num_leaf, dim))
        if self.include_self:
            wh_2d = torch.cat(
                [wh_3d.unsqueeze(-2), wh_2d], dim=-2
            ) # [b, N, 1+num_leaf, d_out]
            h_2d = torch.cat(
                [h_3d.unsqueeze(-2), h_2d], dim=-2
            )

            if self.with_linear_transform:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d)
            else:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d)

            if self.additional:
                h_prime = h_prime + h_3d
        else:
            if self.with_linear_transform:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d) / 2. + wh_3d
            else:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d) / 2. + h_3d

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, wh_2d, wh_3d, num_leaf, include_self=False):
        b, n1, dim = wh_3d.shape
        b, n2, dim = wh_2d.shape

        wh_2d_ = torch.matmul(wh_2d, self.a[:self.out_features, :]) # [b, N2, 1]
        wh_2d_ = torch.reshape(wh_2d_, (b, n1, num_leaf, -1)) # [b, n1, 6, 1]
        wh_3d_ = torch.matmul(wh_3d, self.a[self.out_features:, :]) # [b, N1, 1]

        if include_self:
            wh_2d_ = torch.cat(
                [wh_3d_.unsqueeze(2), wh_2d_], dim=-2
            ) # [b, N1, 1 + num_leaf, 1]

        e = wh_3d_.unsqueeze(2) + wh_2d_
        return self.leakyrelu(e)
