import cv2
import torch
import numpy as np
import os.path as osp
from src.utils.colmap import read_write_model 


def filter_by_track_length(points3D, track_length):
    """ 
    Filter 3d points by track length.
    Return new pcds and corresponding point ids in origin pcds.
    """
    idxs_3d = list(points3D.keys())
    idxs_3d.sort()
    xyzs = np.empty(shape=[0, 3])
    points_idxs = np.empty(shape=[0], dtype=int)
    for i in range(len(idxs_3d)):
        idx_3d = idxs_3d[i]
        if len(points3D[idx_3d].point2D_idxs) < track_length:
            continue
        xyz = points3D[idx_3d].xyz.reshape(1, -1)
        xyzs = np.append(xyzs, xyz, axis=0)
        points_idxs = np.append(points_idxs, idx_3d)
    
    return xyzs, points_idxs


def filter_by_3d_box(points, points_idxs, box_path):
    """ Filter 3d points by 3d box."""
    corner_in_cano = np.loadtxt(box_path)

    assert points.shape[1] == 3, "Input pcds must have shape (n, 3)"
    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, dtype=torch.float32)
    if not isinstance(corner_in_cano, torch.Tensor):
        corner_in_cano = torch.as_tensor(corner_in_cano, dtype=torch.float32)
    
    def filter_(bbox_3d, points):
        """
        @param bbox_3d: corners (8, 3)
        @param points: (n, 3)
        """
        v45 = bbox_3d[5] - bbox_3d[4]
        v40 = bbox_3d[0] - bbox_3d[4]
        v47 = bbox_3d[7] - bbox_3d[4]
        
        points = points - bbox_3d[4]
        m0 = torch.matmul(points, v45)
        m1 = torch.matmul(points, v40)
        m2 = torch.matmul(points, v47)
        
        cs = []
        for m, v in zip([m0, m1, m2], [v45, v40, v47]):
            c0 = 0 < m
            c1 = m < torch.matmul(v, v)
            c = c0 & c1
            cs.append(c)
        cs = cs[0] & cs[1] & cs[2]
        passed_inds = torch.nonzero(cs).squeeze(1)
        num_passed = torch.sum(cs)
        return num_passed, passed_inds, cs
    
    num_passed, passed_inds, keeps = filter_(corner_in_cano, points)
    
    xyzs_filtered = np.empty(shape=(0, 3), dtype=np.float32)
    for i in range(int(num_passed)):
        ind = passed_inds[i]
        xyzs_filtered = np.append(xyzs_filtered, points[ind, None], axis=0)
    
    filtered_xyzs = points[passed_inds]
    passed_inds = points_idxs[passed_inds]
    return filtered_xyzs, passed_inds


def filter_3d(model_path, track_length, box_path):
    """ Filter 3d points by tracke length and 3d box """
    points_model_path = osp.join(model_path, 'points3D.bin')
    points3D = read_write_model.read_points3d_binary(points_model_path)
   
    xyzs, points_idxs = filter_by_track_length(points3D, track_length)
    xyzs, points_idxs = filter_by_3d_box(xyzs, points_idxs, box_path)

    return xyzs, points_idxs
    

def merge(xyzs, points_idxs, dist_threshold=1e-3):
    """ 
    Merge points which are close to others. ({[x1, y1], [x2, y2], ...} => [mean(x_i), mean(y_i)])
    """
    from scipy.spatial.distance import pdist, squareform
    
    if not isinstance(xyzs, np.ndarray):
        xyzs = np.array(xyzs)

    dist = pdist(xyzs, 'euclidean')
    distance_matrix = squareform(dist)
    close_than_thresh = distance_matrix < dist_threshold

    ret_points_count = 0 # num of return points
    ret_points = np.empty(shape=[0, 3]) # pcds after merge
    ret_idxs = {} # {new_point_idx: points idxs in Points3D}

    points3D_idx_record = [] # points that have been merged
    for j in range(distance_matrix.shape[0]):
        idxs = close_than_thresh[j] 

        if np.isin(points_idxs[idxs], points3D_idx_record).any():
            continue

        points = np.mean(xyzs[idxs], axis=0) # new point
        ret_points = np.append(ret_points, points.reshape(1, 3), axis=0)
        ret_idxs[ret_points_count] = points_idxs[idxs]
        ret_points_count += 1
        
        points3D_idx_record = points3D_idx_record + points_idxs[idxs].tolist()
    
    return ret_points, ret_idxs


    