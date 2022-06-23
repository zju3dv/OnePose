import glob
import torch
import hydra
from tqdm import tqdm
import os
import os.path as osp
import natsort

from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils
from src.utils.model_io import load_network
from src.local_feature_2D_detector import LocalFeatureObjectDetector

from pytorch_lightning import seed_everything

seed_everything(12345)


def get_default_paths(cfg, data_root, data_dir, sfm_model_dir):
    anno_dir = osp.join(
        sfm_model_dir, f"outputs_{cfg.network.detection}_{cfg.network.matching}", "anno"
    )
    avg_anno_3d_path = osp.join(anno_dir, "anno_3d_average.npz")
    clt_anno_3d_path = osp.join(anno_dir, "anno_3d_collect.npz")
    idxs_path = osp.join(anno_dir, "idxs.npy")
    sfm_ws_dir = osp.join(
        sfm_model_dir,
        f"outputs_{cfg.network.detection}_{cfg.network.matching}",
        "sfm_ws",
        "model",
    )

    img_lists = []
    color_dir = osp.join(data_dir, "color_full")
    if not osp.exists(color_dir):
        logger.info('color_full directory not exists! Try to parse from Frames.m4v')
        scan_video_dir = osp.join(data_dir, 'Frames.m4v')
        assert osp.exists(scan_video_dir), 'Frames.m4v not found! Run detector fail!'
        data_utils.video2img(scan_video_dir, color_dir)
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)
    
    img_lists = natsort.natsorted(img_lists)

    # Save detect results:
    detect_img_dir = osp.join(data_dir, "color_det")
    if osp.exists(detect_img_dir):
        os.system(f"rm -rf {detect_img_dir}")
    os.makedirs(detect_img_dir, exist_ok=True)

    detect_K_dir = osp.join(data_dir, "intrin_det")
    if osp.exists(detect_K_dir):
        os.system(f"rm -rf {detect_K_dir}")
    os.makedirs(detect_K_dir, exist_ok=True)

    intrin_full_path = osp.join(data_dir, "intrinsics.txt")
    paths = {
        "data_root": data_root,
        "data_dir": data_dir,
        "sfm_model_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "avg_anno_3d_path": avg_anno_3d_path,
        "clt_anno_3d_path": clt_anno_3d_path,
        "idxs_path": idxs_path,
        "intrin_full_path": intrin_full_path,
        "output_detect_img_dir": detect_img_dir,
        "output_K_crop_dir": detect_K_dir
    }
    return img_lists, paths

def load_2D_matching_model(cfg):

    def load_extractor_model(cfg, model_path):
        """Load extractor model(SuperPoint)"""
        from src.models.extractors.SuperPoint.superpoint import SuperPoint
        from src.sfm.extract_features import confs

        extractor_model = SuperPoint(confs[cfg.network.detection]["conf"])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path, force=True)

        return extractor_model

    def load_2D_matcher(cfg):
        """Load matching model(SuperGlue)"""
        from src.models.matchers.SuperGlue.superglue import SuperGlue
        from src.sfm.match_features import confs

        match_model = SuperGlue(confs[cfg.network.matching]["conf"])
        match_model.eval()
        load_network(match_model, cfg.model.match_model_path)
        return match_model

    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    matcher = load_2D_matcher(cfg)
    return extractor_model, matcher


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """Prepare data for OnePose inference"""
    keypoints2d = torch.Tensor(detection["keypoints"])
    descriptors2d = torch.Tensor(detection["descriptors"])

    inp_data = {
        "keypoints2d": keypoints2d[None].cuda(),  # [1, n1, 2]
        "keypoints3d": keypoints3d[None].cuda(),  # [1, n2, 3]
        "descriptors2d_query": descriptors2d[None].cuda(),  # [1, dim, n1]
        "descriptors3d_db": avg_descriptors3d[None].cuda(),  # [1, dim, n2]
        "descriptors2d_db": clt_descriptors[None].cuda(),  # [1, dim, n2*num_leaf]
        "image_size": image_size,
    }

    return inp_data


@torch.no_grad()
def inference_core(cfg, data_root, seq_dir, sfm_model_dir):
    """Inference & visualize"""
    from src.datasets.normalized_dataset import NormalizedDataset
    from src.sfm.extract_features import confs

    # Load models and prepare data:
    extractor_model, matching_2D_model = load_2D_matching_model(cfg)
    img_lists, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)
    K, _ = data_utils.get_K(paths["intrin_full_path"])

    local_feature_obj_detector = LocalFeatureObjectDetector(
        extractor_model,
        matching_2D_model,
        sfm_ws_dir=paths["sfm_ws_dir"],
        n_ref_view=cfg.n_ref_view,
        output_results=True,
        detect_save_dir=paths['output_detect_img_dir'],
        K_crop_save_dir=paths['output_K_crop_dir']
    )
    dataset = NormalizedDataset(
        img_lists, confs[cfg.network.detection]["preprocessing"]
    )
    loader = DataLoader(dataset, num_workers=1)

    # Begin Object detection:
    for id, data in enumerate(tqdm(loader)):
        img_path = data["path"][0]
        inp = data["image"].cuda()

        # Detect object by 2D local feature matching for the first frame:
        local_feature_obj_detector.detect(inp, img_path, K)

def inference(cfg):
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]

    for data_dir, sfm_model_dir in tqdm(
        zip(data_dirs, sfm_model_dirs), total=len(data_dirs)
    ):
        splits = data_dir.split(" ")
        data_root = splits[0]
        for seq_name in splits[1:]:
            seq_dir = osp.join(data_root, seq_name)
            logger.info(f"Run feature matching object detector for: {seq_dir}")
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
