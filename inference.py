import glob
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np

from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils

from pytorch_lightning import seed_everything
seed_everything(12345)


def get_default_paths(cfg, data_root, data_dir, sfm_model_dir):
    anno_dir = osp.join(sfm_model_dir, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
    avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
    clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
    idxs_path = osp.join(anno_dir, 'idxs.npy')

    object_detect_mode = cfg.object_detect_mode
    logger.info(f"Use {object_detect_mode} as object detector")
    if object_detect_mode == 'GT_box':
        color_dir = osp.join(data_dir, 'color')
    elif object_detect_mode == 'feature_matching':
        color_dir = osp.join(data_dir, 'color_det')
        assert osp.exists(color_dir), "color_det directory not exists! You need to run local_feature_2D_detector.py for object detection. Please refer to README.md for the instructions"
    else:
        raise NotImplementedError

    img_lists = []
    img_lists += glob.glob(color_dir + '/*.png', recursive=True)

    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        "data_root": data_root,
        'data_dir': data_dir,
        'sfm_model_dir': sfm_model_dir,
        'avg_anno_3d_path': avg_anno_3d_path,
        'clt_anno_3d_path': clt_anno_3d_path,
        'idxs_path': idxs_path,
        'intrin_full_path': intrin_full_path
    }
    return img_lists, paths


def load_model(cfg):
    """ Load model """
    def load_matching_model(model_path):
        """ Load onepose model """
        from src.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()

        return trained_model

    def load_extractor_model(cfg, model_path):
        """ Load extractor model(SuperPoint) """
        from src.models.extractors.SuperPoint.superpoint import SuperPoint
        from src.sfm.extract_features import confs
        from src.utils.model_io import load_network

        extractor_model = SuperPoint(confs[cfg.network.detection]['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model

    matching_model = load_matching_model(cfg.model.onepose_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return matching_model, extractor_model


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """ Prepare data for OnePose inference """
    keypoints2d = torch.Tensor(detection['keypoints'])
    descriptors2d = torch.Tensor(detection['descriptors'])

    inp_data = {
        'keypoints2d': keypoints2d[None].cuda(), # [1, n1, 2]
        'keypoints3d': keypoints3d[None].cuda(), # [1, n2, 3]
        'descriptors2d_query': descriptors2d[None].cuda(), # [1, dim, n1]
        'descriptors3d_db': avg_descriptors3d[None].cuda(), # [1, dim, n2]
        'descriptors2d_db': clt_descriptors[None].cuda(), # [1, dim, n2*num_leaf]
        'image_size': image_size
    }

    return inp_data


@torch.no_grad()
def inference_core(cfg, data_root, seq_dir, sfm_model_dir):
    """ Inference & visualize"""
    from src.datasets.normalized_dataset import NormalizedDataset
    from src.sfm.extract_features import confs
    from src.evaluators.cmd_evaluator import Evaluator

    matching_model, extractor_model = load_model(cfg)
    img_lists, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)

    dataset = NormalizedDataset(img_lists, confs[cfg.network.detection]['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)
    evaluator = Evaluator()

    idx = 0
    num_leaf = cfg.num_leaf
    avg_data = np.load(paths['avg_anno_3d_path'])
    clt_data = np.load(paths['clt_anno_3d_path'])
    idxs = np.load(paths['idxs_path'])

    keypoints3d = torch.Tensor(clt_data['keypoints3d']).cuda()
    num_3d = keypoints3d.shape[0]
    # Load average 3D features:
    avg_descriptors3d, _ = data_utils.pad_features3d_random(
                                avg_data['descriptors3d'],
                                avg_data['scores3d'],
                                num_3d
                            )
    # Load corresponding 2D features of each 3D point:
    clt_descriptors, _ = data_utils.build_features3d_leaves(
                                clt_data['descriptors3d'],
                                clt_data['scores3d'],
                                idxs, num_3d, num_leaf
                            )

    for data in tqdm(loader):
        img_path = data['path'][0]
        inp = data['image'].cuda()

        intrin_path = path_utils.get_intrin_path_by_color(img_path, det_type=cfg.object_detect_mode)
        K_crop = np.loadtxt(intrin_path)

        # Detect query image keypoints and extract descriptors:
        pred_detection = extractor_model(inp)
        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        # 2D-3D matching by GATsSPG:
        inp_data = pack_data(avg_descriptors3d, clt_descriptors, 
                             keypoints3d, pred_detection, data['size'])
        pred, _ = matching_model(inp_data)
        matches = pred['matches0'].detach().cpu().numpy()
        valid = matches > -1
        kpts2d = pred_detection['keypoints']
        kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
        confidence = pred['matching_scores0'].detach().cpu().numpy()
        mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]

        # Estimate object pose by 2D-3D correspondences:
        pose_pred, pose_pred_homo, inliers = eval_utils.ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)

        # Evaluate:
        gt_pose_path = path_utils.get_gt_pose_path_by_color(img_path, det_type=cfg.object_detect_mode)
        pose_gt = np.loadtxt(gt_pose_path)
        evaluator.evaluate(pose_pred, pose_gt)

        # Visualize:
        if cfg.save_wis3d:
            poses = [pose_gt, pose_pred_homo]
            box3d_path = path_utils.get_3d_box_path(data_root)
            intrin_full_path = path_utils.get_intrin_full_path(seq_dir)
            image_full_path = path_utils.get_img_full_path_by_color(img_path, det_type=cfg.object_detect_mode)

            image_full = vis_utils.vis_reproj(image_full_path, poses, box3d_path, intrin_full_path,
                                              save_demo=cfg.save_demo, demo_root=cfg.demo_root)
            mkpts3d_2d = vis_utils.reproj(K_crop, pose_gt, mkpts3d)
            image0 = Image.open(img_path).convert('LA')
            image1 = image0.copy()
            vis_utils.dump_wis3d(idx, cfg, seq_dir, image0, image1, image_full,
                                 mkpts2d, mkpts3d_2d, mconf, inliers)

            idx += 1

    eval_result = evaluator.summarize()
    obj_name = sfm_model_dir.split('/')[-1]
    seq_name = seq_dir.split('/')[-1]
    eval_utils.record_eval_result(cfg.output.eval_dir, obj_name, seq_name, eval_result)


def inference(cfg):
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]

    for data_dir, sfm_model_dir in tqdm(zip(data_dirs, sfm_model_dirs), total=len(data_dirs)):
        splits = data_dir.split(" ")
        data_root = splits[0]
        for seq_name in splits[1:]:
            seq_dir = osp.join(data_root, seq_name)
            logger.info(f'Eval {seq_dir}')
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)


@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg):
    globals()[cfg.type](cfg)

if __name__ == "__main__":
    main()
