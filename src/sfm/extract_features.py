import h5py
import tqdm
import torch
import logging

from torch.utils.data import DataLoader

confs = {
    'superpoint': {
        'output': 'feats-spp',
        'model': {
            'name': 'spp_det',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_h': 512,
            'resize_w': 512
        },
        'conf': {
            'descriptor_dim': 256,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'keypoints_threshold': 0.6
        }
    }
}


@torch.no_grad()
def spp(img_lists, feature_out, cfg):
    """extract keypoints info by superpoint"""
    from src.utils.model_io import load_network
    from src.models.extractors.SuperPoint.superpoint import SuperPoint as spp_det
    from src.datasets.normalized_dataset import NormalizedDataset
    
    conf = confs[cfg.network.detection]
    model = spp_det(conf['conf']).cuda()
    model.eval()
    load_network(model, cfg.network.detection_model_path, force=True)

    dataset = NormalizedDataset(img_lists, conf['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    for data in tqdm.tqdm(loader):
        inp = data['image'].cuda()
        pred = model(inp)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = data['size'][0].numpy()
        
        grp = feature_file.create_group(data['path'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
        
        del pred
    
    feature_file.close()
    logging.info('Finishing exporting features.')


def main(img_lists, feature_out, cfg):
    if cfg.network.detection == 'superpoint':
        spp(img_lists, feature_out, cfg)
    else:
        raise NotImplementedError