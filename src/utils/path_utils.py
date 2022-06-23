import os
import os.path as osp


"""
For each object, we store in the following directory format:

data_root:
    - box3d_corners.txt
    - seq1_root
        - intrinsics.txt
        - color/
        - color_det[optional]/
        - poses_ba/
        - intrin_ba/
        - intrin_det[optional]/
        - ......
    - seq2_root
    - ......
"""

def get_gt_pose_path_by_color(color_path, det_type='GT_box'):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/poses_ba/").replace(
            ".png", ".txt"
        )
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/poses_ba/").replace(
            ".png", ".txt"
        )
    else:
        raise NotImplementedError

def get_img_full_path_by_color(color_path, det_type='GT_box'):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/color_full/")
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/color_full/")
    else:
        raise NotImplementedError

def get_intrin_path_by_color(color_path, det_type='GT_box'):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/intrin_ba/").replace(
            ".png", ".txt"
        )
    elif det_type == 'feature_matching':
        return color_path.replace("/color_det/", "/intrin_det/").replace(
            ".png", ".txt"
        )
    else:
        raise NotImplementedError

def get_intrin_dir(seq_root):
    return osp.join(seq_root, "intrin_ba")

def get_gt_pose_dir(seq_root):
    return osp.join(seq_root, "poses_ba")

def get_intrin_full_path(seq_root):
    return osp.join(seq_root, "intrinsics.txt")

def get_3d_box_path(data_root):
    return osp.join(data_root, "box3d_corners.txt")

