import h5py
import json
import os.path as osp
import numpy as np

from collections import defaultdict
from pathlib import Path

from src.utils.colmap import read_write_model
from src.utils import path_utils


def get_default_path(cfg, outputs_dir):
    """
    directory tree:

                outputs_dir
                    |
            ------------------
            |                |
        anno_dir           deep_sfm_dir
                             |
                         ---------
                         |       |
                   database.db  model
    """
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    model_dir = osp.join(deep_sfm_dir, 'model')
    anno_dir = osp.join(outputs_dir, 'anno')
    
    Path(anno_dir).mkdir(exist_ok=True, parents=True)

    return model_dir, anno_dir


def inverse_id_name(images):
    """ traverse keys of images.bin({id: image_name}), get {image_name: id} mapping. """
    inverse_dict = {}
    for key in images.keys():
        img_name = images[key].name
        
        inverse_dict[img_name] = key

    return inverse_dict


def id_mapping(points_idxs):
    """ traverse points_idxs({new_3dpoint_id: old_3dpoints_idxs}), get {old_3dpoint_idx: new_3dpoint_idx} mapping. """
    kp3d_id_mapping = {} # {old_point_idx: new_point_idx}
    
    for new_point_idx, old_point_idxs in points_idxs.items():
        for old_point_idx in old_point_idxs:
            assert old_point_idx not in kp3d_id_mapping.keys() 
            kp3d_id_mapping[old_point_idx] = new_point_idx
    
    return kp3d_id_mapping


def gather_3d_anno(keypoints_2d, descriptors_2d, scores_2d, kp3d_idxs, image_name,
                   feature_idxs, kp3d_id_position, kp3d_id_feature, kp3d_id_score, kp3d_id_image):
    """ For each 3d point, gather all corresponding 2d information """
    kp3d_idx_to_kp2d_idx = {}
    for kp3d_idx, feature_idx in zip(kp3d_idxs, feature_idxs):
        kp3d_idx_to_kp2d_idx[kp3d_idx] = feature_idx
        
        if kp3d_idx not in kp3d_id_feature:
            kp3d_id_position[kp3d_idx] = keypoints_2d[feature_idx][None]
            kp3d_id_feature[kp3d_idx] = descriptors_2d[:, feature_idx][None]
            kp3d_id_score[kp3d_idx] = scores_2d[feature_idx][None]
            kp3d_id_image[kp3d_idx] = [image_name]
        else:
            kp3d_id_position[kp3d_idx] = np.append(kp3d_id_position[kp3d_idx],
                                                   keypoints_2d[feature_idx][None], 
                                                   axis=0)
            kp3d_id_feature[kp3d_idx] = np.append(kp3d_id_feature[kp3d_idx],
                                                  descriptors_2d[:, feature_idx][None],
                                                  axis=0)
            kp3d_id_score[kp3d_idx] = np.append(kp3d_id_score[kp3d_idx],
                                                scores_2d[feature_idx][None],
                                                axis=0)
            kp3d_id_image[kp3d_idx].append(image_name)

    return kp3d_id_position, kp3d_id_feature, kp3d_id_score, kp3d_id_image, kp3d_idx_to_kp2d_idx


def read_features(feature):
    """ decouple keypoints, descriptors and scores from feature """
    keypoints_2d = feature['keypoints'].__array__()
    descriptors_2d = feature['descriptors'].__array__()
    scores_2d = feature['scores'].__array__()

    return keypoints_2d, descriptors_2d, scores_2d
 

def count_features(img_lists, features, images, kp3d_id_mapping):
    """ Search for valid 2d-3d correspondences; Count 3d features. """
    kp3d_idx_position = {} # {new_3d_point_idx: [position1, position2, ...]}
    kp3d_idx_feature = {} # {new_3d_point_idx: [feature1, feature2, ...]}
    kp3d_idx_score = {} # {new_3d_point_idx: [score1, score2, ...]}
    kp3d_idx_image = {} # {new_3d_point_idx: [image1, image2, ...]}
    kp3d_idx_to_img_kp2d_idx = defaultdict(dict) # {new_3d_point_idx: {image_name: 2d_point_idx}}

    inverse_dict = inverse_id_name(images) # {image_name: id}
    # traverse each image to find valid 2d-3d correspondence
    for img_name in img_lists:
        feature = features[img_name]
        keypoints_2d, descriptors_2d, scores_2d = read_features(feature)
        feature_dim = descriptors_2d.shape[0]
        
        id_ = inverse_dict[img_name]
        image_info = images[id_]
        point3D_ids = image_info.point3D_ids

        filter_feature_idxs = [] # record valid 2d point idxs. Each of these 2d points have a correspondence with a 3d point.
        filter_kp3d_idxs = [] # record valid 3d point idxs. Each of these 3d points have a correspondence with a 2d point in this image.
        feature_idxs = np.where(point3D_ids != -1)[0] if np.any(point3D_ids != -1) else None
        if feature_idxs is None:
            kp3d_idx_to_img_kp2d_idx[img_name] = {}
        else:
            for feature_idx in feature_idxs:
                kp3d_idx = point3D_ids[feature_idx]
                if kp3d_idx in kp3d_id_mapping.keys(): # select 3d points which are kept after filter
                    filter_kp3d_idxs.append(kp3d_idx)
                    filter_feature_idxs.append(feature_idx)
            
            kp3d_idx_position, kp3d_idx_feature, kp3d_idx_score, kp3d_idx_image, kp3d_idx_to_kp2d_idx = \
                gather_3d_anno(
                    keypoints_2d, descriptors_2d, scores_2d,
                    filter_kp3d_idxs, img_name, filter_feature_idxs,
                    kp3d_idx_position, kp3d_idx_feature, kp3d_idx_score, kp3d_idx_image
                )

            kp3d_idx_to_img_kp2d_idx[img_name] = kp3d_idx_to_kp2d_idx
    
    return feature_dim, kp3d_idx_position, kp3d_idx_feature, kp3d_idx_score, kp3d_idx_image, kp3d_idx_to_img_kp2d_idx


def average_3d_ann(kp3d_id_feature, kp3d_id_score, xyzs, points_idxs, feature_dim):
    """ 
    average position, descriptors and scores for 3d points 
    new_point_feature = avg(all merged 3d points features) = avg(all matched 2d points features)
    """
    kp3d_descriptors = np.empty(shape=(0, feature_dim))
    kp3d_scores = np.empty(shape=(0, 1))
    kp3d_position = np.empty(shape=(0, 3))

    for new_point_idx, old_points_idxs in points_idxs.items():
        descriptors = np.empty(shape=(0, feature_dim))
        scores = np.empty(shape=(0, 1))
        for old_point_idx in old_points_idxs:
            descriptors = np.append(descriptors, kp3d_id_feature[old_point_idx], axis=0)
            scores = np.append(scores, kp3d_id_score[old_point_idx].reshape(-1, 1), axis=0)
        
        avg_descriptor = np.mean(descriptors, axis=0).reshape(1, -1)
        avg_score = np.mean(scores, axis=0).reshape(1, -1)
        
        kp3d_position = np.append(kp3d_position, xyzs[new_point_idx].reshape(1, 3), axis=0)
        
        kp3d_descriptors = np.append(kp3d_descriptors, avg_descriptor, axis=0)
        kp3d_scores = np.append(kp3d_scores, avg_score, axis=0)
    
    return kp3d_position, kp3d_descriptors, kp3d_scores 


def gather_3d_ann(kp3d_id_position, kp3d_id_feature, kp3d_id_score, kp3d_id_image, xyzs, points_idxs, feature_dim):
    """ 
    Gather affiliated 2d points' positions, (mean/concated)descriptors and scores for each 3d points
    """
    kp3d_descriptors = np.empty(shape=(0, feature_dim))
    kp3d_scores = np.empty(shape=(0, 1))
    kp3d_position = np.empty(shape=(0, 3))
    idxs = []

    for new_point_idx, old_points_idxs in points_idxs.items():
        descriptors = np.empty(shape=(0, feature_dim))
        scores = np.empty(shape=(0, 1))
        affi_position = np.empty(shape=(0, 2))
        for old_point_idx in old_points_idxs:
            descriptors = np.append(descriptors, kp3d_id_feature[old_point_idx], axis=0)
            scores = np.append(scores, kp3d_id_score[old_point_idx].reshape(-1, 1), axis=0)
            affi_position = np.append(affi_position, kp3d_id_position[old_point_idx], axis=0)

        kp3d_position = np.append(kp3d_position, xyzs[new_point_idx].reshape(1, 3), axis=0) 
        kp3d_descriptors = np.append(kp3d_descriptors, descriptors, axis=0)
        kp3d_scores = np.append(kp3d_scores, scores, axis=0)
        idxs.append(descriptors.shape[0])
    
    return kp3d_position, kp3d_descriptors, kp3d_scores, np.array(idxs)


def save_3d_anno(xyzs, descriptors, scores, out_path):
    """ Save 3d anno for each object """
    descriptors = descriptors.transpose(1, 0)
    np.savez(out_path, keypoints3d=xyzs, descriptors3d=descriptors, scores3d=scores)
 
 
def get_assign_matrix(xys, xyzs, kp3d_idx_to_kp2d_idx, kp3d_id_mapping):
    """ 
    Given 2d-3d correspondence(n pairs), build assign matrix(2*n) for this image 
    @param xys: all 2d keypoints extracted in this image.
    @param xyzs: all 3d points after filter.
    @param kp3d_idx_to_kp2d_idx: valid 2d-3d correspondences in this image. {kp3d_idx: kp2d_idx}
    @param kp3d_id_mapping: {3dpoint_before_filter_idx: 3dpoint_after_filter_idx}
    """
    kp2d_idxs = np.arange(len(xys))
    kp3d_idxs = np.arange(len(xyzs))

    MN1 = []
    for idx3d, idx2d in kp3d_idx_to_kp2d_idx.items():
        assert idx3d in kp3d_id_mapping.keys()
        new_idx3d = kp3d_id_mapping[idx3d]
        new_idx2d = idx2d

        if new_idx3d not in kp3d_idxs:
            kp2d_idxs = np.delete(kp2d_idxs, np.where(kp2d_idxs == new_idx2d))
            continue

        assert new_idx2d in kp2d_idxs and new_idx3d in kp3d_idxs
        kp2d_idxs = np.delete(kp2d_idxs, np.where(kp2d_idxs == new_idx2d)) 
        kp3d_idxs = np.delete(kp3d_idxs, np.where(kp3d_idxs == new_idx3d))
        
        MN1.append([new_idx2d, new_idx3d])

    num_matches = len(MN1)
    assign_matrix = np.array(MN1).T

    print("=> match pairs num: ", num_matches)
    print('=> total 2d points: ', xys.shape[0])
    print("=> total 3d points: ", xyzs.shape[0])
    return num_matches, assign_matrix


def save_2d_anno_for_each_image(cfg, img_path, keypoints_2d, descriptors_2d, scores_2d, 
                                assign_matrix, num_matches):
    """ save annotation for each image
                   data_dir
                      |
            --------------------------
            |         |      ...     |
           color     anno    ...    pose
            |         |              |
        image_name  anno_file       pose_file    
    """
    data_dir = osp.dirname(osp.dirname(img_path))
    anno_dir = osp.join(data_dir, f'anno_{cfg.network.detection}')
    Path(anno_dir).mkdir(exist_ok=True, parents=True)

    img_name = osp.basename(img_path)
    anno_2d_path = osp.join(anno_dir, img_name.replace('.png', '.json'))

    anno_2d = {
        'keypoints2d': keypoints_2d.tolist(), # [n, 2]
        'descriptors2d': descriptors_2d.tolist(), # [dim, n]
        'scores2d': scores_2d.reshape(-1, 1).tolist(), # [n, 1]
        'assign_matrix': assign_matrix.tolist(), # [2, k]
        'num_matches': num_matches
    }
    
    with open(anno_2d_path, 'w') as f:
        json.dump(anno_2d, f)    
    
    return anno_2d_path


def save_2d_anno(cfg, img_lists, features, filter_xyzs, points_idxs, 
                 kp3d_idx_to_img_kp2d_idx, anno2d_out_path):
    """ Save 2d annotations for each image and gather all 2d annotations """
    annotations = []
    anno_id = 0
    
    kp3d_id_mapping = id_mapping(points_idxs)

    for img_path in img_lists:
        feature = features[img_path]
        kp3d_idx_to_kp2d_idx = kp3d_idx_to_img_kp2d_idx[img_path]
        
        keypoints_2d, descriptors_2d, scores_2d = read_features(feature) 
        num_matches, assign_matrix = get_assign_matrix(
            keypoints_2d, filter_xyzs,
            kp3d_idx_to_kp2d_idx, kp3d_id_mapping
        )

        if num_matches != 0:
            anno_2d_path = save_2d_anno_for_each_image(cfg, img_path, keypoints_2d, descriptors_2d, scores_2d, assign_matrix, num_matches) 
            pose_path = path_utils.get_gt_pose_path_by_color(img_path)
            anno_id += 1
            annotation = {
                'anno_id': anno_id, 'anno_file': anno_2d_path,
                'img_file': img_path, 'pose_file': pose_path
            }
            annotations.append(annotation)
    
    with open(anno2d_out_path, 'w') as f:
        json.dump(annotations, f)
 

def mean_descriptors(descriptors, idxs):
    """ Average leaf nodes' descriptors"""
    cumsum_idxs = np.cumsum(idxs)
    pre_cumsum_idxs = np.cumsum(idxs)[:-1]
    pre_cumsum_idxs = np.insert(pre_cumsum_idxs, 0, 0)

    descriptors_instance = [np.mean(descriptors[start: end], axis=0).reshape(1, -1) for start, end in zip(pre_cumsum_idxs, cumsum_idxs)]
    avg_descriptors = np.concatenate(descriptors_instance, axis=0)
    return avg_descriptors


def mean_scores(scores, idxs):
    """ Average leaf nodes' scores"""
    cumsum_idxs = np.cumsum(idxs)
    pre_cumsum_idxs = np.cumsum(idxs)[:-1]
    pre_cumsum_idxs = np.insert(pre_cumsum_idxs, 0, 0)

    scores_instance = [np.mean(scores[start: end], axis=0).reshape(1, -1)
                        for start, end in zip(pre_cumsum_idxs, cumsum_idxs)]
    avg_scores = np.concatenate(scores_instance, axis=0)
    return avg_scores


def get_kpt_ann(cfg, img_lists, feature_file_path, outputs_dir, points_idxs, xyzs):
    """ Generate 3d point feature.
    @param xyzs: 3d points after filter(track length, 3d box and merge operation)
    @param points_idxs: {new_point_id: [old_point1_id, old_point2_id, ...]}.
                        new_point_id: [0, xyzs.shape[0])
                        old_point_id*: point idx in Points3D.bin
                        This param is used to record the relationship of points after filter and before filter.

    Main idea:
        1. Concat features for each 3d point;
            2d_point_a ... 2d_point_*  ...  2d_point_b ... 2d_point_*
                    \     /                        \     /
                    3d_point1          ...       3d_point_*
                        \              merge        /
                         \             ...         /
                               new point feature
        2. Average features;
        3. Generate assign matrix;
        4. Save annotations;
    """
    model_dir, anno_out_dir = get_default_path(cfg, outputs_dir)

    cameras, images, points3D = read_write_model.read_model(model_dir, ext='.bin')
    features = h5py.File(feature_file_path, 'r')

    kp3d_id_mapping = id_mapping(points_idxs)
    feature_dim, kp3d_id_position, kp3d_id_feature, kp3d_id_score, kp3d_id_image, kp3d_idx_to_img_kp2d_idx \
        = count_features(img_lists, features, images, kp3d_id_mapping)

    filter_xyzs,  filter_descriptors, filter_scores, idxs = gather_3d_ann(kp3d_id_position, kp3d_id_feature, kp3d_id_score, kp3d_id_image, xyzs,
                                                                                                                points_idxs, feature_dim)
    
    avg_descriptors, avg_scores = mean_descriptors(filter_descriptors, idxs), mean_scores(filter_scores, idxs)

    anno2d_out_path = osp.join(anno_out_dir, 'anno_2d.json')
    save_2d_anno(cfg, img_lists, features, filter_xyzs, points_idxs, kp3d_idx_to_img_kp2d_idx, anno2d_out_path)
    
    avg_anno3d_out_path = osp.join(anno_out_dir, 'anno_3d_average.npz')
    collect_anno3d_out_path = osp.join(anno_out_dir, 'anno_3d_collect.npz')
    save_3d_anno(filter_xyzs, avg_descriptors, avg_scores, avg_anno3d_out_path)
    save_3d_anno(filter_xyzs, filter_descriptors, filter_scores, collect_anno3d_out_path)

    idxs_out_path = osp.join(anno_out_dir, 'idxs.npy')
    np.save(idxs_out_path, idxs)
