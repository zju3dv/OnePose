import os.path as osp
import time
import cv2
import torch
import numpy as np
from src.utils.colmap.read_write_model import read_model
from src.utils.data_utils import get_K_crop_resize, get_image_crop_resize
from src.utils.vis_utils import reproj


def pack_extract_data(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    image = image[None] / 255.0
    return torch.Tensor(image)


def pack_match_data(db_detection, query_detection, db_size, query_size):
    data = {}
    for k in db_detection.keys():
        data[k + "0"] = db_detection[k].__array__()
    for k in query_detection.keys():
        data[k + "1"] = query_detection[k].__array__()
    data = {k: torch.from_numpy(v)[None].float().cuda() for k, v in data.items()}

    data["image0"] = torch.empty(
        (
            1,
            1,
        )
        + tuple(db_size)[::-1]
    )
    data["image1"] = torch.empty(
        (
            1,
            1,
        )
        + tuple(query_size)[::-1]
    )
    return data


class LocalFeatureObjectDetector():
    def __init__(self, extractor, matcher, sfm_ws_dir, n_ref_view=15, output_results=False, detect_save_dir=None, K_crop_save_dir=None):
        self.extractor = extractor.cuda()
        self.matcher = matcher.cuda()
        self.db_dict = self.extract_ref_view_features(sfm_ws_dir, n_ref_view)
        self.output_results = output_results
        self.detect_save_dir = detect_save_dir
        self.K_crop_save_dir = K_crop_save_dir

    def extract_ref_view_features(self, sfm_ws_dir, n_ref_views):
        assert osp.exists(sfm_ws_dir), f"SfM work space:{sfm_ws_dir} not exists!"
        cameras, images, points3D = read_model(sfm_ws_dir)
        idx = 0
        sample_gap = len(images) // n_ref_views

        # Prepare reference input to matcher:
        db_dict = {}  # id: image
        for idx in range(1, len(images), sample_gap):
            db_img_path = images[idx].name

            db_img = pack_extract_data(db_img_path)

            # Detect DB image keypoints:
            db_inp = db_img[None].cuda()
            db_detection = self.extractor(db_inp)
            db_detection = {
                k: v[0].detach().cpu().numpy() for k, v in db_detection.items()
            }
            db_detection["size"] = np.array(db_img.shape[-2:])
            db_dict[idx] = db_detection

        return db_dict

    @torch.no_grad()
    def match_worker(self, query):
        detect_results_dict = {}
        for idx, db in self.db_dict.items():
            db_shape = db["size"]
            query_shape = query["size"]

            match_data = pack_match_data(db, query, db["size"], query["size"])
            match_pred = self.matcher(match_data)
            matches = match_pred["matches0"][0].detach().cpu().numpy()
            confs = match_pred["matching_scores0"][0].detach().cpu().numpy()
            valid = matches > -1

            mkpts0 = db["keypoints"][valid]
            mkpts1 = query["keypoints"][matches[valid]]
            confs = confs[valid]

            if mkpts0.shape[0] < 6:
                affine = None
                inliers = np.empty((0))
                detect_results_dict[idx] = {
                    "inliers": inliers,
                    "bbox": np.array([0, 0, query["size"][0], query["size"][1]]),
                }
                continue

            # Estimate affine and warp source image:
            affine, inliers = cv2.estimateAffinePartial2D(
                mkpts0, mkpts1, ransacReprojThreshold=6
            )

            # Estimate box:
            four_corner = np.array(
                [
                    [0, 0, 1],
                    [db_shape[1], 0, 1],
                    [0, db_shape[0], 1],
                    [db_shape[1], db_shape[0], 1],
                ]
            ).T  # 3*4

            bbox = (affine @ four_corner).T.astype(np.int32)  # 4*2

            left_top = np.min(bbox, axis=0)
            right_bottom = np.max(bbox, axis=0)

            w, h = right_bottom - left_top
            offset_percent = 0.0
            x0 = left_top[0] - int(w * offset_percent)
            y0 = left_top[1] - int(h * offset_percent)
            x1 = right_bottom[0] + int(w * offset_percent)
            y1 = right_bottom[1] + int(h * offset_percent)

            detect_results_dict[idx] = {
                "inliers": inliers,
                "bbox": np.array([x0, y0, x1, y1]),
            }
        return detect_results_dict

    def detect_by_matching(self, query):
        detect_results_dict = self.match_worker(query)

        # Sort multiple bbox candidate and use bbox with maxium inliers:
        idx_sorted = [
            k
            for k, _ in sorted(
                detect_results_dict.items(),
                reverse=True,
                key=lambda item: item[1]["inliers"].shape[0],
            )
        ]
        return detect_results_dict[idx_sorted[0]]["bbox"]

    def robust_crop(self, query_img_path, bbox, K, crop_size=512):
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]
        x_c = (x0 + x1) / 2
        y_c = (y0 + y1) / 2

        origin_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
        image_crop = origin_img
        K_crop, K_crop_homo = get_K_crop_resize(bbox, K, [crop_size, crop_size])
        return image_crop, K_crop

    def crop_img_by_bbox(self, query_img_path, bbox, K=None, crop_size=512):
        """
        Crop image by detect bbox
        Input:
            query_img_path: str,
            bbox: np.ndarray[x0, y0, x1, y1],
            K[optional]: 3*3
        Output:
            image_crop: np.ndarray[crop_size * crop_size],
            K_crop[optional]: 3*3
        """
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]
        origin_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)

        resize_shape = np.array([y1 - y0, x1 - x0])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox, K, resize_shape)
        image_crop, trans1 = get_image_crop_resize(origin_img, bbox, resize_shape)

        bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([crop_size, crop_size])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox_new, K_crop, resize_shape)
        image_crop, trans2 = get_image_crop_resize(image_crop, bbox_new, resize_shape)
        
        return image_crop, K_crop if K is not None else None
    
    def save_detection(self, crop_img, query_img_path):
        if self.output_results and self.detect_save_dir is not None:
            cv2.imwrite(osp.join(self.detect_save_dir, osp.basename(query_img_path)), crop_img)
    
    def save_K_crop(self, K_crop, query_img_path):
        if self.output_results and self.K_crop_save_dir is not None:
            np.savetxt(osp.join(self.K_crop_save_dir, osp.splitext(osp.basename(query_img_path))[0] + '.txt'), K_crop) # K_crop: 3*3

    def detect(self, query_img, query_img_path, K, crop_size=512):
        """
        Detect object by local feature matching and crop image.
        Input:
            query_image: np.ndarray[1*1*H*W],
            query_img_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        if len(query_img.shape) != 4:
            query_inp = query_img[None].cuda()
        else:
            query_inp = query_img.cuda()
        
        # Extract query image features:
        query_inp = self.extractor(query_inp)
        query_inp = {k: v[0].detach().cpu().numpy() for k, v in query_inp.items()}
        query_inp["size"] = np.array(query_img.shape[-2:])

        # Detect bbox and crop image:
        bbox = self.detect_by_matching(
            query=query_inp,
        )
        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self.save_detection(image_crop, query_img_path)
        self.save_K_crop(K_crop, query_img_path)

        # To Tensor:
        image_crop = image_crop.astype(np.float32) / 255
        image_crop_tensor = torch.from_numpy(image_crop)[None][None].cuda()

        return bbox, image_crop_tensor, K_crop
    
    def previous_pose_detect(self, query_img_path, K, pre_pose, bbox3D_corner, crop_size=512):
        """
        Detect object by projecting 3D bbox with estimated last frame pose.
        Input:
            query_image_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
            pre_pose: np.ndarray[3*4] or [4*4], pose of last frame
            bbox3D_corner: np.ndarray[8*3], corner coordinate of annotated 3D bbox
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        # Project 3D bbox:
        proj_2D_coor = reproj(K, pre_pose, bbox3D_corner)
        x0, y0 = np.min(proj_2D_coor, axis=0)
        x1, y1 = np.max(proj_2D_coor, axis=0)
        bbox = np.array([x0, y0, x1, y1]).astype(np.int32)

        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self.save_detection(image_crop, query_img_path)
        self.save_K_crop(K_crop, query_img_path)

        # To Tensor:
        image_crop = image_crop.astype(np.float32) / 255
        image_crop_tensor = torch.from_numpy(image_crop)[None][None].cuda()

        return bbox, image_crop_tensor, K_crop