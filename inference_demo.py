import glob
import torch
import hydra
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import natsort

from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils
from src.utils.model_io import load_network
from src.local_feature_2D_detector import LocalFeatureObjectDetector

from src.tracker.ba_tracker import BATracker

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
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)
    
    img_lists = natsort.natsorted(img_lists)

    # Visualize detector:
    vis_detector_dir = osp.join(data_dir, "detector_vis")
    if osp.exists(vis_detector_dir):
        os.system(f"rm -rf {vis_detector_dir}")
    os.makedirs(vis_detector_dir, exist_ok=True)
    det_box_vis_video_path = osp.join(data_dir, "det_box.mp4")

    # Visualize pose:
    vis_box_dir = osp.join(data_dir, "pred_vis")
    if osp.exists(vis_box_dir):
        os.system(f"rm -rf {vis_box_dir}")
    os.makedirs(vis_box_dir, exist_ok=True)
    demo_video_path = osp.join(data_dir, "demo_video.mp4")

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
        "vis_box_dir": vis_box_dir,
        "vis_detector_dir": vis_detector_dir,
        "det_box_vis_video_path": det_box_vis_video_path,
        "demo_video_path": demo_video_path,
    }
    return img_lists, paths


def load_model(cfg):
    """Load model"""

    def load_matching_model(model_path):
        """Load onepose model"""
        from src.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()

        return trained_model

    def load_extractor_model(cfg, model_path):
        """Load extractor model(SuperPoint)"""
        from src.models.extractors.SuperPoint.superpoint import SuperPoint
        from src.sfm.extract_features import confs

        extractor_model = SuperPoint(confs[cfg.network.detection]["conf"])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path, force=True)

        return extractor_model

    matching_model = load_matching_model(cfg.model.onepose_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return matching_model, extractor_model


def load_2D_matching_model(cfg):
    def load_2D_matcher(cfg):
        from src.models.matchers.SuperGlue.superglue import SuperGlue
        from src.sfm.match_features import confs

        match_model = SuperGlue(confs[cfg.network.matching]["conf"])
        match_model.eval()
        load_network(match_model, cfg.model.match_model_path)
        return match_model

    matcher = load_2D_matcher(cfg)
    return matcher


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


def inference_core(cfg, data_root, seq_dir, sfm_model_dir):
    """Inference & visualize"""
    from src.datasets.normalized_dataset import NormalizedDataset
    from src.sfm.extract_features import confs
    if cfg.use_tracking:
        logger.warning("The tracking module is under development. "
                       "Running OnePose inference without tracking instead.")
        tracker = BATracker(cfg)
        track_interval = 5
    else:
        logger.info("Running OnePose inference without tracking")

    # Load models and prepare data:
    matching_model, extractor_model = load_model(cfg)
    matching_2D_model = load_2D_matching_model(cfg)
    img_lists, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)

    # sort images
    im_ids = [int(osp.basename(i).replace('.png', '')) for i in img_lists]
    im_ids.sort()
    img_lists = [osp.join(osp.dirname(img_lists[0]), f'{im_id}.png') for im_id in im_ids]

    K, _ = data_utils.get_K(paths["intrin_full_path"])
    box3d_path = path_utils.get_3d_box_path(data_root)
    bbox3d = np.loadtxt(box3d_path)

    local_feature_obj_detector = LocalFeatureObjectDetector(
        extractor_model,
        matching_2D_model,
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=False,
        detect_save_dir=paths["vis_detector_dir"],
    )
    dataset = NormalizedDataset(
        img_lists, confs[cfg.network.detection]["preprocessing"]
    )
    loader = DataLoader(dataset, num_workers=1)

    # Prepare 3D features:
    num_leaf = cfg.num_leaf
    avg_data = np.load(paths["avg_anno_3d_path"])
    clt_data = np.load(paths["clt_anno_3d_path"])
    idxs = np.load(paths["idxs_path"])

    keypoints3d = torch.Tensor(clt_data["keypoints3d"]).cuda()
    num_3d = keypoints3d.shape[0]
    # load average 3D features:
    avg_descriptors3d, _ = data_utils.pad_features3d_random(
        avg_data["descriptors3d"], avg_data["scores3d"], num_3d
    )
    # load corresponding 2D features of each 3D point:
    clt_descriptors, _ = data_utils.build_features3d_leaves(
        clt_data["descriptors3d"], clt_data["scores3d"], idxs, num_3d, num_leaf
    )

    pred_poses = {}  # {id:[pred_pose, inliers]}

    for id, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            img_path = data["path"][0]
            inp = data["image"].cuda()

            # Detect object:
            if id == 0:
                # Detect object by 2D local feature matching for the first frame:
                bbox, inp_crop, K_crop = local_feature_obj_detector.detect(inp, img_path, K)
            else:
                # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
                previous_frame_pose, inliers = pred_poses[id - 1]

                if len(inliers) < 8:
                    # Consider previous pose estimation failed, reuse local feature object detector:
                    bbox, inp_crop, K_crop = local_feature_obj_detector.detect(
                        inp, img_path, K
                    )
                else:
                    (
                        bbox,
                        inp_crop,
                        K_crop,
                    ) = local_feature_obj_detector.previous_pose_detect(
                        img_path, K, previous_frame_pose, bbox3d
                    )

            # Detect query image(cropped) keypoints and extract descriptors:
            pred_detection = extractor_model(inp_crop)
            pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

            # 2D-3D matching by GATsSPG:
            inp_data = pack_data(
                avg_descriptors3d,
                clt_descriptors,
                keypoints3d,
                pred_detection,
                data["size"],
            )
            pred, _ = matching_model(inp_data)
            matches = pred["matches0"].detach().cpu().numpy()
            valid = matches > -1
            kpts2d = pred_detection["keypoints"]
            kpts3d = inp_data["keypoints3d"][0].detach().cpu().numpy()
            confidence = pred["matching_scores0"].detach().cpu().numpy()
            mkpts2d, mkpts3d, mconf = (
                kpts2d[valid],
                kpts3d[matches[valid]],
                confidence[valid],
            )

            # Estimate object pose by 2D-3D correspondences:
            pose_pred, pose_pred_homo, inliers = eval_utils.ransac_PnP(
                K_crop, mkpts2d, mkpts3d, scale=1000
            )

            # Store previous estimated poses:
            pred_poses[id] = [pose_pred, inliers]
            image_crop = np.asarray((inp_crop * 255).squeeze().cpu().numpy(), dtype=np.uint8)

        if cfg.use_tracking:
            frame_dict = {
                'im_path': image_crop,
                'kpt_pred': pred_detection,
                'pose_pred': pose_pred_homo,
                'pose_gt': pose_pred_homo,
                'K': K_crop,
                'K_crop': K_crop,
                'data': data
            }

            use_update = id % track_interval == 0
            if use_update:
                mkpts3d_db_inlier = mkpts3d[inliers.flatten()]
                mkpts2d_q_inlier = mkpts2d[inliers.flatten()]

                n_kpt = kpts2d.shape[0]

                valid_query_id = np.where(valid != False)[0][inliers.flatten()]
                kpts3d_full = np.ones([n_kpt, 3]) * 10086
                kpts3d_full[valid_query_id] = mkpts3d_db_inlier
                kpt3d_ids = matches[valid][inliers.flatten()]

                kf_dict = {
                    'im_path': image_crop,
                    'kpt_pred': pred_detection,
                    'valid_mask': valid,
                    'mkpts2d': mkpts2d_q_inlier,
                    'mkpts3d': mkpts3d_db_inlier,
                    'kpt3d_full': kpts3d_full,
                    'inliers': inliers,
                    'kpt3d_ids': kpt3d_ids,

                    'valid_query_id': valid_query_id,
                    'pose_pred': pose_pred_homo,
                    'pose_gt': pose_pred_homo,
                    'K': K_crop
                }

                need_update = not tracker.update_kf(kf_dict)

            if id == 0:
                tracker.add_kf(kf_dict)
                id += 1
                pose_opt = pose_pred_homo
            else:
                pose_init, pose_opt, ba_log = tracker.track(frame_dict, auto_mode=False)
        else:
            pose_opt = pose_pred_homo

        # Visualize:
        vis_utils.save_demo_image(
            pose_opt,
            K,
            image_path=img_path,
            box3d_path=box3d_path,
            draw_box=len(inliers) > 6,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

    # Output video to visualize estimated poses:
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])


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
            logger.info(f"Eval {seq_dir}")
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
