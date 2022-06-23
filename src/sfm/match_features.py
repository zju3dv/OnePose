import h5py
import torch
import logging
import tqdm

import os.path as osp

confs = {
    'superglue': {
        'output': 'matches-spg',
        'conf': {
            'descriptor_dim': 256,
            'weights': 'outdoor',
            'match_threshold': 0.7
        }
    }
}


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


@torch.no_grad()
def spg(cfg, feature_path, covis_pairs, matches_out, vis_match=False):
    """Match features by SuperGlue"""
    from src.models.matchers.SuperGlue.superglue import SuperGlue as spg_matcher
    from src.utils.model_io import load_network
    from src.utils.vis_utils import vis_match_pairs
    
    assert osp.exists(feature_path), feature_path
    feature_file = h5py.File(feature_path, 'r')
    logging.info(f'Exporting matches to {matches_out}')
    
    with open(covis_pairs, 'r') as f:
        pair_list = f.read().rstrip('\n').split('\n')
    
    # load superglue model
    conf = confs[cfg.network.matching]['conf']
    model = spg_matcher(conf).cuda()
    model.eval()
    load_network(model, cfg.network.matching_model_path, force=True)

    # match features by superglue
    match_file = h5py.File(matches_out, 'w')
    matched = set()
    for pair in tqdm.tqdm(pair_list):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        if len({(name0, name1), (name1, name0)} & matched) \
            or pair in match_file:
            continue
        
        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats0.keys():
            data[k+'0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k+'1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().cuda() for k, v in data.items()}
        
        data['image0'] = torch.empty((1, 1, ) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1, ) + tuple(feats1['image_size'])[::-1])
        pred = model(data)

        grp = match_file.create_group(pair)
        matches0 = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches0)
        
        matches1 = pred['matches1'][0].cpu().short().numpy()
        grp.create_dataset('matches1', data=matches1)
        
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)
        
        if 'matching_scores1' in pred:
            scores = pred['matching_scores1'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores1', data=scores)
        
        matched |= {(name0, name1), (name1, name0)}
        
        if vis_match:
            vis_match_pairs(pred, feats0, feats1, name0, name1)
    
    match_file.close()
    logging.info('Finishing exporting matches.')


def main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False):
    if cfg.network.matching == 'superglue':
        spg(cfg, feature_out, covis_pairs_out, matches_out, vis_match)
    else:
        raise NotImplementedError