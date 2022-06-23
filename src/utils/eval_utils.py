import cv2
import numpy as np
import os.path as osp

from pathlib import Path

def record_eval_result(out_dir, obj_name, seq_name, eval_result):
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    out_file = osp.join(out_dir, obj_name + seq_name + '.txt')
    f = open(out_file, 'w')
    for k, v in eval_result.items():
        f.write(f'{k}: {v}\n')

    f.close()


def ransac_PnP(K, pts_2d, pts_3d, scale=1):
    """ solve pnp """
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')
    
    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64)) 
    K = K.astype(np.float64)
    
    pts_3d *= scale
    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs, reprojectionError=5,
                                                    iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)

        rotation = cv2.Rodrigues(rvec)[0]

        tvec /= scale
        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []


def query_pose_error(pose_pred, pose_gt):
    """
    Input:
    ---------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


def compute_query_pose_errors(data, preds):
    query_pose_gt = data['query_pose_gt'][0].cpu().numpy()
    query_K = data['query_intrinsic'][0].cpu().numpy()
    query_kpts2d = data['keypoints2d'][0].cpu().numpy()
    query_kpts3d = data['keypoints3d'][0].cpu().numpy()

    matches0 = preds['matches0'].cpu().numpy()
    confidence = preds['matching_scores0'].cpu().numpy()
    valid = matches0 > -1
    mkpts2d = query_kpts2d[valid]
    mkpts3d = query_kpts3d[matches0[valid]]
    mconf = confidence[valid]

    pose_pred = []
    val_results = {'R_errs': [], 't_errs': [], 'inliers': []}

    query_pose_pred, query_pose_pred_homo, inliers = ransac_PnP(
        query_K,
        mkpts2d,
        mkpts3d
    )
    pose_pred.append(query_pose_pred_homo)

    if query_pose_pred is None:
        val_results['R_errs'].append(np.inf)
        val_results['t_errs'].append(np.inf)
        val_results['inliers'].append(np.array([])).astype(np.bool)
    else:
        R_err, t_err = query_pose_error(query_pose_pred, query_pose_gt)
        val_results['R_errs'].append(R_err)
        val_results['t_errs'].append(t_err)
        val_results['inliers'].append(inliers)
    
    pose_pred = np.stack(pose_pred)

    val_results.update({'mkpts2d': mkpts2d, 'mkpts3d': mkpts3d, 'mconf': mconf}) 
    return pose_pred, val_results


def aggregate_metrics(metrics, thres=[1, 3, 5]):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    R_errs = metrics['R_errs']
    t_errs = metrics['t_errs']
    
    degree_distance_metric = {}
    for threshold in thres:
        degree_distance_metric[f'{threshold}cm@{threshold}degree'] = np.mean(
            (np.array(R_errs) < threshold) & (np.array(t_errs) < threshold)
        )
    
    return degree_distance_metric