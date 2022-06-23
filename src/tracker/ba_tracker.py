import cv2
import numpy as np
import torch
from DeepLM.TorchLM.solver import Solve

from src.models.matchers.nn.nearest_neighbour import NearestNeighbour
from src.utils.eval_utils import ransac_PnP
from src.tracker.tracking_utils import Timer, project, SnavelyReprojectionErrorV2, MovieWriter


device = 'cuda'


class BATracker:
    def __init__(self, cfg):
        self.kf_frames = dict()
        self.query_frames = dict()
        self.id = 0
        self.last_kf_id = -1
        self.vis = None
        self.extractor = self.load_extractor_model(cfg, cfg.model.extractor_model_path)
        self.matcher = NearestNeighbour()
        self.pose_list = []

        self.kpt2ds = []  # coordinate for kpt
        self.kpt2d_available_list = []
        self.kpt2d_descs = []  # may be change to descriptor list
        self.kpt2d_fids = []  # fid for kpt
        self.cams = []  # list of cam params
        self.kf_kpt_index_dict = dict()  # kf_id -> [2d_id_start, 2d_id_end]
        self.db_3d_list = np.array([])

        self.kpt3d_list = []
        self.kpt3d_sourceid = []
        self.kpt2d3d_ids = []  # 3D ids of each 2D keypoint
        self.update_th = 10
        self.frame_id = 0
        self.last_kf_info = None
        # self.win_size = 30 # for static scene
        self.win_size = 10  # for dynamic scene
        self.frame_interval = 5
        self.mw = MovieWriter()
        self.out = './track_kpt.mp4'
        self.init_cnt = 3
        self.use_motion_cnt = 0
        self.tm = Timer()

    def reset(self):
        self.kf_frames = dict()
        self.query_frames = dict()
        self.id = 0
        self.last_kf_id = -1
        self.pose_list = []

        self.kpt2ds = []  # coordinate for kpt
        self.kpt2d_available_list = []
        self.kpt2d_descs = []  # may be change to descriptor list
        self.kpt2d_fids = []  # fid for kpt
        self.cams = []  # list of cam params
        self.kf_kpt_index_dict = dict()  # kf_id -> [2d_id_start, 2d_id_end]
        self.db_3d_list = np.array([])

        self.kpt3d_list = []
        self.kpt3d_sourceid = []
        self.kpt2d3d_ids = []  # 3D ids of each 2D keypoint
        self.update_th = 10
        self.frame_id = 0
        self.last_kf_info = None
        self.win_size = 20  # for static scene
        # self.win_size = 10 # for dynamic scene
        self.frame_interval = 3
        # from src.utils.movie_writer import MovieWriter
        # self.mw = MovieWriter()
        # self.out = './track_kpt.mp4'
        self.init_cnt = 3
        self.use_motion_cnt = 0
        # from src.tracker.vis_utils import Timer
        # self.tm = Timer()

    def load_extractor_model(self, cfg, model_path):
        """ Load extractor model(SuperGlue) """
        from src.sfm.extract_features import confs
        from src.utils.model_io import load_network
        from src.models.extractors.SuperPoint.superpoint import SuperPoint

        extractor_model = SuperPoint(confs[cfg.network.detection]['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def cm_degree_5_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        return translation_distance, angular_distance

    def kpt_flow_track(self, im_kf, im_query, kpt2d_last):
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # kpt_last = np.array(, dtype=np.float32)
        kpt_last = np.expand_dims(kpt2d_last, axis=1)
        kpt_new, status, err = cv2.calcOpticalFlowPyrLK(im_kf, im_query, kpt_last, None, **lk_params)
        if status is not None:
            valid_id = np.where(status.flatten() == 1)
            kpt_new = np.squeeze(kpt_new, axis=1)
            return kpt_new, valid_id
        else:
            return None, None

    def update_kf(self, kf_info_dict):
        if self.last_kf_info is not None:
            pose_last = self.pose_list[-1]
            trans_dist, rot_dist = self.cm_degree_5_metric(kf_info_dict['pose_pred'], pose_last)
            # [Option] Reject invalid updates
            if trans_dist > 10 or rot_dist > 10:
                trans_dist_gt, rot_dist_gt = self.cm_degree_5_metric(kf_info_dict['pose_pred'], kf_info_dict['pose_gt'])
                print(f"Update rejected:{trans_dist}/{rot_dist} - {trans_dist_gt}/{rot_dist_gt}")
                return False
            else:
                self.last_kf_info = kf_info_dict
                return True
            # self.last_kf_info = kf_info_dict
            # return True
        else:

            self.last_kf_info = kf_info_dict
            return True

    def add_kf(self, kf_info_dict):
        self.kf_frames[self.id] = kf_info_dict
        self.pose_list.append(kf_info_dict['pose_pred'])

        if len(self.kpt2ds) == 0:
            # update camera params
            self.cams = np.array([self.get_cam_param(kf_info_dict['K'], kf_info_dict['pose_pred'])])

            # initialize 2D keypoints
            kpt_pred = kf_info_dict['kpt_pred']

            n_kpt = kpt_pred['keypoints'].shape[0]
            self.kpt2ds = kpt_pred['keypoints']  # [n_2d, 2]
            self.kpt2d_match = np.zeros([n_kpt], dtype=int)  # [n_2d, ]
            self.kpt2d_descs = kpt_pred['descriptors'].transpose()  # [n_2d, n_dim]
            self.kpt2d_fids = np.ones([n_kpt], dtype=int) * self.id

            # initialize camera_list
            self.kf_kpt_index_dict[self.id] = (0, n_kpt - 1)

            # init 3D points & 2D-3D relationship
            self.kpt3d_list = np.array(kf_info_dict['mkpts3d'])
            self.kpt3d_source_id = np.array(np.ones([len(self.kpt3d_list)], dtype=int) * self.id)
            self.kpt2d3d_ids = np.ones([n_kpt], dtype=int) * -1

            kf_3d_ids = np.arange(0, len(kf_info_dict['mkpts3d']))
            self.kpt2d3d_ids[kf_info_dict['valid_query_id']] = kf_3d_ids

            kf_db_ids = kf_info_dict['kpt3d_ids']
            # create mapping from DB id to kpt3d id
            self.db_3d_list = kf_db_ids
        else:
            # update camera params
            kf_cam = np.array([self.get_cam_param(kf_info_dict['K'], kf_info_dict['pose_pred'])])
            self.cams = np.concatenate([self.cams, kf_cam], axis=0)

            # update 2D keypoints
            kpt_pred = kf_info_dict['kpt_pred']
            n_kpt = kpt_pred['keypoints'].shape[0]
            self.kpt2ds = np.concatenate([self.kpt2ds, kpt_pred['keypoints']], axis=0)  # [n_2d, 2]
            self.kpt2d_match = np.concatenate([self.kpt2d_match, np.zeros([n_kpt])], axis=0)  # [n_2d, ]
            self.kpt2d_descs = np.concatenate([self.kpt2d_descs, kpt_pred['descriptors'].transpose()],
                                              axis=0)  # [n_2d, ]
            self.kpt2d_fids = np.concatenate([self.kpt2d_fids, np.ones([n_kpt]) * self.id])

            # initialize camera_list
            start_id = self.kf_kpt_index_dict[self.last_kf_id][-1] + 1
            self.kf_kpt_index_dict[self.id] = (start_id, start_id + n_kpt - 1)

            # Find non-duplicate 3d ids in kf 3d points and in database 3d points
            kpt3d_db_ids = self.db_3d_list
            kf_db_ids = kf_info_dict['kpt3d_ids']
            # duplicate 3D keypoints in kf db_ids and current frmae
            intersect_kpts = np.intersect1d(kpt3d_db_ids, kf_db_ids)
            mask_kf_3d_exist = np.in1d(kf_db_ids, intersect_kpts)  # [bool, ]
            mask_db3d_exist = np.in1d(kpt3d_db_ids, intersect_kpts)  # [bool, ]

            # kf_3d_ids_ndup = np.where(mask_kf_3d_exist == False)[0] # non-duplicate kf 3d keypoint ids
            kf_kpt3ds_new = kf_info_dict['mkpts3d'][np.where(mask_kf_3d_exist == False)[0]]  # get new 3D keypionts

            # Update 2D-3D relationship
            kf_kpt2d3d_id = np.ones([n_kpt]) * -1
            valid_2d_id = kf_info_dict['valid_query_id']

            # For duplicate parts, 3D ids copy from existing ids
            valid_id_dup = valid_2d_id[np.where(mask_kf_3d_exist == True)[0]]
            kf_kpt2d3d_id[valid_id_dup] = \
                np.arange(0, len(self.kpt3d_list))[np.where(mask_db3d_exist == True)[0]]  # index on existing 3D db id

            # For non-duplicate parts, 3D ids are created
            kpt3d_start_id = len(self.kpt3d_list)
            valid_id_ndup = valid_2d_id[np.where(mask_kf_3d_exist == False)[0]]
            kf_kpt2d3d_id[valid_id_ndup] = np.arange(kpt3d_start_id, kpt3d_start_id + len(kf_kpt3ds_new))
            kf_kpt2d3d_id = np.asarray(kf_kpt2d3d_id, dtype=int)
            self.kpt2d3d_ids = np.concatenate([self.kpt2d3d_ids, kf_kpt2d3d_id], axis=0)

            # Update 3D keypoints
            self.kpt3d_list = np.concatenate([self.kpt3d_list, kf_kpt3ds_new], axis=0)
            self.kpt3d_source_id = np.concatenate([self.kpt3d_list, np.ones(kf_kpt3ds_new.shape[0],
                                                                            dtype=int) * self.id])

            # update mapping from DB id to kpt3d id
            kf_db_ids_new = kf_db_ids[valid_id_ndup]  # non-duplicate kf 3d keypoint db id
            self.db_3d_list = np.concatenate([self.db_3d_list, kf_db_ids_new])
        # TODO: examine result after update

        self.last_kf_info = kf_info_dict
        self.last_kf_id = self.id
        self.id += 1

    def cuda2cpu(self, pred_detection_cuda):
        return {k: v[0].cpu().numpy() for k, v in pred_detection_cuda.items()}

    def apply_match(self, kpt_pred0, kpt_pred1):
        import torch
        data = {}
        for k in kpt_pred0.keys():
            data[k + '0'] = kpt_pred0[k]
        for k in kpt_pred1.keys():
            data[k + '1'] = kpt_pred1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device) for k, v in data.items()}
        matching_result = self.matcher(data)
        return matching_result

    def _triangulatePy(self, P1, P2, kpt2d_1, kpt2d_2):
        import scipy.linalg
        point3d = []
        for p1, p2 in zip(kpt2d_1, kpt2d_2):
            A = np.zeros([4, 4])
            A[0, :] = p1[0] * P1[2, :] - P1[0, :]
            A[1, :] = p1[1] * P1[2, :] - P1[1, :]
            A[2, :] = p2[0] * P2[2, :] - P2[0, :]
            A[3, :] = p2[1] * P2[2, :] - P2[1, :]

            U, a, Vh = scipy.linalg.svd(np.dot(A.T, A))
            X = -1 * U[:, 3]
            point3d.append(X[:3] / X[3])

        return np.array(point3d)

    def apply_triangulation(self, K1, K2, Tcw1, Tcw2, kpt2d_1, kpt2d_2):
        proj_mat1 = np.dot(K1, np.linalg.inv(Tcw1)[:3, :])
        proj_mat2 = np.dot(K2, np.linalg.inv(Tcw2)[:3, :])
        point_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, kpt2d_1.transpose(), kpt2d_2.transpose()).T
        point_3d_w = point_4d[:, :3] / np.repeat(point_4d[:, 3], 3).reshape(-1, 3)
        # point_3d_w2 = triangulatePy(proj_mat1, proj_mat2, kpt2d_1, kpt2d_2)
        return point_3d_w

    def motion_prediction(self):
        from transforms3d.euler import mat2euler, euler2mat
        pose0 = self.pose_list[-3]
        pose1 = self.pose_list[-2]
        pose_t = self.pose_list[-1]

        speed_trans = ((pose1[:3, 3] - pose0[:3, 3]) + (pose_t[:3, 3] - pose1[:3, 3])) / 2

        rot0 = np.array(mat2euler(pose0[:3, :3]))
        rot1 = np.array(mat2euler(pose1[:3, :3]))
        rot_t = np.array(mat2euler(pose1[:3, :3]))
        speed_rot = ((rot1 - rot0) + (rot_t - rot1)) / 2
        trans_t = pose_t[:3, 3] + speed_trans
        rot_t = rot_t + speed_rot

        pose_new = np.eye(4)
        pose_new[:3, :3] = euler2mat(rot_t[0], rot_t[1], rot_t[2])
        pose_new[:3, 3] = trans_t
        return pose_new

    def flow_track(self, frame_info_dict, kf_frame_info):
        from src.tracker.tracking_utils import draw_kpt2d, put_text

        self.tm.tick('0-1-load image')

        im_kf = kf_frame_info['im_path']
        im_query = frame_info_dict['im_path']
        self.tm.tock('0-1-load image')

        self.tm.tick('0-2-flow track')
        mkpts2d_query, valid_ids = self.kpt_flow_track(im_kf, im_query, kf_frame_info['mkpts2d'])
        self.tm.tock('0-2-flow track')

        # from matplotlib import pyplot as plt
        # plt.imshow(im_kf)
        # plt.savefig(f'./vis/{self.frame_id}_0.png')
        # plt.close()
        #
        # plt.imshow(im_query)
        # plt.savefig(f'./vis/{self.frame_id}_1.png')
        # plt.close()

        if valid_ids is not None:
            self.tm.tick('0-3-set info')
            kpt3ds_kf = kf_frame_info['mkpts3d'][valid_ids]
            mkpts2d_query = mkpts2d_query[valid_ids]
            # Solve PnP to find initial pose
            pose_init, pose_init_homo, inliers = ransac_PnP(frame_info_dict['K_crop'], mkpts2d_query, kpt3ds_kf)
            self.tm.tock('0-3-set info')

            self.tm.tick('0-4-cal dist')
            kpt3d_rep = project(kpt3ds_kf, frame_info_dict['K_crop'], pose_init)
            kpt_dist = np.mean(np.linalg.norm(kpt3d_rep - mkpts2d_query, axis=1))
            trans_dist, rot_dist = self.cm_degree_5_metric(pose_init_homo, frame_info_dict['pose_gt'])
            self.tm.tock('0-4-cal dist')

            print(f"\nFlow pose error:{trans_dist} - {rot_dist}\nKptDist:{kpt_dist}")

            # FIXME: draw flow tracking result for debug
            label = f"Flow pose error:{trans_dist} - {rot_dist}"
            im_query_vis = cv2.cvtColor(im_query, cv2.COLOR_GRAY2RGB)
            im_out = draw_kpt2d(im_query_vis, mkpts2d_query)
            im_out = put_text(im_out, label)
            scale = 1
            h, w, c = im_out.shape
            h_res = int(h / scale)
            w_res = int(w / scale)
            im_out = cv2.resize(im_out, (w_res, h_res))
            self.mw.write(im_out, self.out)

        else:
            kpt_dist = 1000086
            trans_dist = kpt_dist
            rot_dist = kpt_dist
            pose_init_homo = None

        if trans_dist <= 7 and rot_dist <= 7 and kpt_dist < 10:
            frame_info_dict['mkpts3d'] = kpt3ds_kf
            frame_info_dict['mkpts2d'] = mkpts2d_query
            self.last_kf_info = frame_info_dict

        return pose_init_homo, kpt_dist

    def apply_ba(self, kpt2ds, kpt2d3d_ids, kpt2d_fids, kpt3d_list, cams, verbose=False):
        device = 'cuda'
        points = torch.tensor(kpt3d_list, device=device, dtype=torch.float64, requires_grad=False)
        cam_pose = torch.tensor(cams[:, :6], device=device, dtype=torch.float64, requires_grad=False)
        valid2d_idx = np.where(kpt2d3d_ids != -1)[0]
        if np.max(kpt2d_fids) > self.win_size:
            valid_idx_fid = np.where(kpt2d_fids > np.max(kpt2d_fids) - self.win_size)
            valid2d_idx = np.intersect1d(valid_idx_fid, valid2d_idx)
        ks = cams[np.array(kpt2d_fids[valid2d_idx], dtype=int), 6:]
        features = torch.tensor(np.concatenate([kpt2ds[valid2d_idx], ks], axis=1), device=device, dtype=torch.float64,
                                requires_grad=False)
        ptIdx = torch.tensor(kpt2d3d_ids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)
        camIdx = torch.tensor(kpt2d_fids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)

        if verbose:
            # Display Initial Reprojection Error by frame
            kpt2ds_np = kpt2ds[valid2d_idx]
            kpt3d_idx = kpt2d3d_ids[valid2d_idx]
            camera_idx = np.asarray(kpt2d_fids[valid2d_idx], dtype=int)
            kpt3ds = kpt3d_list
            cams_np = cams[camera_idx]
            for frame_idx in np.unique(camera_idx):
                kpt_idx = np.where(camera_idx == frame_idx)[0]
                # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
                kps2d = kpt2ds_np[kpt_idx]
                kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
                kps_cam = cams_np[kpt_idx]
                K, pose_mat = self.get_cam_params_back(kps_cam[0])
                kps_rep = project(kps3d, K, pose_mat[:3])
                kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
                print(f'Frame:{frame_idx} with {len(kps2d)} kpts\n'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}\n'
                      f'- sum:{np.sum(kps_rep_error)}')
        points, cam_pose, features, ptIdx, camIdx = points.to(device), \
                                                    cam_pose.to(device), features.to(device), \
                                                    ptIdx.to(device), camIdx.to(device)

        if device == 'cuda':
            torch.cuda.synchronize()

        # optimize
        Solve(variables=[points, cam_pose],
              constants=[features],
              indices=[ptIdx, camIdx],
              fn=SnavelyReprojectionErrorV2,
              numIterations=5,
              numSuccessIterations=5,
              verbose=verbose)

        points_opt_np = points.cpu().detach().numpy()
        cam_opt_np = cam_pose.cpu().detach().numpy()

        # Display Optimized Reprojection Error by frame
        if verbose:
            kpt2ds_np = kpt2ds[valid2d_idx]
            kpt3d_idx = kpt2d3d_ids[valid2d_idx]
            camera_idx = np.asarray(kpt2d_fids[valid2d_idx], dtype=int)
            kpt3ds = points_opt_np
            cams_np = cam_opt_np[camera_idx]
            cams_K_np = cams[camera_idx, 6:]
            for frame_idx in np.unique(camera_idx):
                kpt_idx = np.where(camera_idx == frame_idx)[0]
                # print(len(kpt_idx))
                # print(kpt_idx[:10])
                # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
                kps2d = kpt2ds_np[kpt_idx]
                kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
                kps_cam = cams_np[kpt_idx]
                kps_cam_K = cams_K_np[kpt_idx]
                kps_cam_input = np.concatenate([kps_cam[0], kps_cam_K[0]])
                K, pose_mat = self.get_cam_params_back(kps_cam_input)
                kps_rep = project(kps3d, K, pose_mat[:3])
                kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
                print(f'Frame:{frame_idx}\n'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}\n'
                      f'- sum:{np.sum(kps_rep_error)}')
        # points_opt_np = points.cpu().numpy()
        # cam_opt_np = cameras.cpu().numpy()
        cam_opt = np.concatenate([cam_opt_np, cams[:, 6:]], axis=1)
        return points_opt_np, cam_opt

    def get_cam_params_back(self, cam_params):
        """ Convert BAL format to frame parameter to matrix form"""
        r_vec = cam_params[:3]
        t = cam_params[3:6]
        f = cam_params[6]
        k1 = cam_params[7]
        k2 = cam_params[8]
        K = np.array(
            [[f, 0, k1],
             [0, f, k2],
             [0, 0, 1]])
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = cv2.Rodrigues(r_vec)[0]
        pose_mat[:3, 3] = t
        return K, pose_mat

    def get_cam_param(self, K, pose):
        """ Convert frame parameter to BAL format"""
        f = K[0, 0]
        k1 = K[0, 2]
        k2 = K[1, 2]
        t = pose[:3, 3]
        R = cv2.Rodrigues(pose[:3, :3])[0]
        return np.concatenate([R.flatten(), t, [f, k1, k2]])

    def track_ba(self, frame_info_dict, verbose=True):
        print(f"Updating frame id:{self.frame_id} [WIN SIZE:{len(np.unique(self.kpt2d_fids))}]")

        ba_log = dict()
        pose_init = frame_info_dict['pose_init']
        # Load image
        self.tm.tick('1-load image')
        kf_frame_info = self.kf_frames[self.last_kf_id]
        im_kf = kf_frame_info['im_path']

        self.tm.tock('1-load image')

        kpt2ds_pred_query = frame_info_dict['kpt_pred']
        kpt2ds_pred_query.pop('scores')

        # Get KF 2D keypoints from data
        self.tm.tick('2-extract kpt')
        kpt_idx_start, kpt_idx_end = self.kf_kpt_index_dict[self.last_kf_id]
        kpt_idx = np.arange(kpt_idx_start, kpt_idx_end + 1)
        kpt2ds_pred_kf = \
            {'keypoints': self.kpt2ds[kpt_idx],
             'descriptors': self.kpt2d_descs[kpt_idx].transpose()}
        # kpt2ds_pred_kf = kf_frame_info['kpt_pred']
        self.tm.tock('2-extract kpt')

        # Apply match
        self.tm.tick('3-match kpt')
        T_0to1 = np.dot(np.linalg.inv(kf_frame_info['pose_gt']), frame_info_dict['pose_gt'])
        print(f"Input kf:{len(kpt2ds_pred_kf['keypoints'])} - {len(kpt2ds_pred_query['keypoints'])}")
        match_results = self.apply_match(kpt2ds_pred_kf, kpt2ds_pred_query)

        match_kq = match_results['matches0'][0].cpu().numpy()
        valid = np.where(match_kq != -1)
        mkpts2d_kf = kpt2ds_pred_kf['keypoints'][valid]
        mkpts2d_query = kpt2ds_pred_query['keypoints'][match_kq[valid]]
        kpt_idx_valid = kpt_idx[valid]
        self.tm.tock('3-match kpt')

        # Update
        self.tm.tick('5-update 2d3d ids')
        # kpt2ds_match_f = np.copy(self.kpt2d_match)
        kpt2d3d_ids_f = np.copy(self.kpt2d3d_ids)

        # Update 2D inform
        n_kpt_q = len(mkpts2d_query)
        kpt2ds_f = np.concatenate([self.kpt2ds, mkpts2d_query])  # update 2D keypoints
        # kpt2ds_match_f[valid] += 1 # update 2D match
        # kpt2ds_match_f = np.concatenate([kpt2ds_match_f, np.ones([n_kpt_q])])

        # Check 2D-3D correspondence
        kf_2d_3d_ids = kpt2d3d_ids_f[kpt_idx_valid]
        kpt_idx_wo3d = np.where(kf_2d_3d_ids == -1)[0]  # local index of point without 3D index
        mkpts2d_kf_triang = mkpts2d_kf[kpt_idx_wo3d]
        mkpts2d_query_triang = mkpts2d_query[kpt_idx_wo3d]

        # Update 2D-3D correspondence for existing points
        query_2d3d_ids = np.ones(n_kpt_q) * -1
        # estimate reprojection error
        kpt_idx_w3d = np.where(kf_2d_3d_ids != -1)[0]
        print(f"Found {len(kpt_idx_w3d)} points with known 3D")
        ba_log['pt_found'] = len(kpt_idx_w3d)

        mkps3d_query_exist = self.kpt3d_list[kf_2d_3d_ids[kpt_idx_w3d]]
        mkpts2d_query_exist = mkpts2d_query[kpt_idx_w3d]
        kpt2d_rep_query_exist = project(mkps3d_query_exist, frame_info_dict['K'], pose_init[:3])
        rep_diff_q_exist = np.linalg.norm(kpt2d_rep_query_exist - mkpts2d_query_exist, axis=1)

        # [Option] keep only points with reprojection error smaller than 20
        med = np.median(rep_diff_q_exist) * 1.2
        exist_keep_idx_q_rep = np.where(rep_diff_q_exist < med)[0]

        exist_keep_idx_q = exist_keep_idx_q_rep
        kpt_idx_w3d_keep = kpt_idx_w3d[exist_keep_idx_q]
        print(f"Known point removed:{len(np.where(rep_diff_q_exist >= 20)[0])}")

        query_2d3d_ids[kpt_idx_w3d_keep] = kf_2d_3d_ids[kpt_idx_w3d_keep]
        self.tm.tock('5-update 2d3d ids')

        # Triangulation
        self.tm.tick('6-triangulation')
        if len(kpt_idx_wo3d) > 0:
            Tco_kf = np.linalg.inv(kf_frame_info['pose_pred'])
            Tco_query = np.linalg.inv(pose_init)
            kpt3ds_triang = self.apply_triangulation(kf_frame_info['K'],
                                                     frame_info_dict['K'],
                                                     Tco_kf, Tco_query,
                                                     mkpts2d_kf_triang, mkpts2d_query_triang)

            # Remove triangulation points with extremely large error
            kpt2d_rep_kf = project(kpt3ds_triang, kf_frame_info['K'], kf_frame_info['pose_pred'][:3])
            kpt2d_rep_query = project(kpt3ds_triang, frame_info_dict['K'], pose_init[:3])
            rep_diff_q = np.linalg.norm(kpt2d_rep_query - mkpts2d_query_triang, axis=1)
            rep_diff_kf = np.linalg.norm(kpt2d_rep_kf - mkpts2d_kf_triang, axis=1)
            # Remove 3D points with large error
            triang_rm_idx_q = np.where(rep_diff_q > 20)[0]
            triang_rm_idx_kf = np.where(rep_diff_kf > 20)[0]

            # [Option] Remove 3D points distant away
            triang_rm_idx_dist = np.where(kpt3ds_triang[:, 2] > 0.15)[0]
            triang_rm_idx = np.unique(np.concatenate([triang_rm_idx_q, triang_rm_idx_kf, triang_rm_idx_dist]))
            # triang_rm_idx = np.unique(np.concatenate([triang_rm_idx_q, triang_rm_idx_kf]))

            triang_keep_idx = np.array([i for i in range(len(kpt2d_rep_query))
                                        if i not in triang_rm_idx])  # index over mkpts2d_q_triang

            print(f"{len(triang_rm_idx)} removed out of {len(kpt_idx_wo3d)} points")
            ba_log['pt_triang'] = len(kpt_idx_wo3d)
            ba_log['pt_triang_rm'] = len(triang_rm_idx)

            if len(triang_keep_idx) != 0:
                mkpts2d_kf_triang = mkpts2d_kf_triang[triang_keep_idx]
                mkpts2d_query_triang = mkpts2d_query_triang[triang_keep_idx]
                kpt2d_rep_kf = kpt2d_rep_kf[triang_keep_idx]
                kpt2d_rep_query = kpt2d_rep_query[triang_keep_idx]
        else:
            triang_keep_idx = []
        self.tm.tock('6-triangulation')

        self.tm.tick('7-update triangulation')
        # Update correspondence for newly triangulated points
        if len(kpt_idx_wo3d) > 0 and len(triang_keep_idx) > 0:
            kpt3d_start_id = len(self.kpt3d_list)
            query_2d3d_ids[kpt_idx_wo3d[triang_keep_idx]] = np.arange(kpt3d_start_id, kpt3d_start_id
                                                                      + len(triang_keep_idx))

        query_2d3d_ids = np.asarray(query_2d3d_ids, dtype=int)
        kpt2d3d_ids_f = np.concatenate([self.kpt2d3d_ids, query_2d3d_ids])

        # Add 3D points
        if len(kpt_idx_wo3d) > 0 and len(triang_keep_idx) > 0:
            kpt3d_list_f = np.concatenate([self.kpt3d_list, kpt3ds_triang[triang_keep_idx]])
        else:
            kpt3d_list_f = np.copy(self.kpt3d_list)
        cams_f = np.concatenate([self.cams, [self.get_cam_param(frame_info_dict['K'], pose_init)]])
        kpt2d_fids_f = np.concatenate([self.kpt2d_fids, np.ones([n_kpt_q]) * self.id])

        n_triang_pt = len(triang_keep_idx)
        if verbose:
            # ###################  Calculate Reprojection Error and visualization  ############################
            kpt_idxs = np.where(kpt2d_fids_f == np.max(kpt2d_fids_f))[0]
            start_idx = np.min(kpt_idxs)
            kpt_idxs = kpt_idxs[np.where(kpt2d3d_ids_f[kpt_idxs] != -1)[0]]
            if len(kpt_idxs) != 0:
                # print(len(kpt_idxs))
                # print(kpt_idxs[:10])
                kpt3d_full = kpt3d_list_f[kpt2d3d_ids_f[kpt_idxs]]
                kpt2d_full = kpt2ds_f[kpt_idxs]
                # print(kpt2d3d_ids_f[kpt_idxs][:10])
                # rep3d_full = project(kpt3d_full, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
                rep3d_full = project(kpt3d_full, frame_info_dict['K'], pose_init[:3])
                kps_error_full = np.linalg.norm(kpt2d_full - rep3d_full, axis=1)
                print(f'Full points: {len(kps_error_full)}'
                      f'- min:{np.min(kps_error_full)}\n'
                      f'- max:{np.max(kps_error_full)}\n'
                      f'- med:{np.median(kps_error_full)}')

            if n_triang_pt > 0:
                kps2d_triang_ids = np.where(kpt2d3d_ids_f[start_idx:] >= len(self.kpt3d_list))[0] + start_idx
                kpt3d_triang_ids = kpt2d3d_ids_f[kps2d_triang_ids]
                rep3d = project(kpt3d_list_f[kpt3d_triang_ids], frame_info_dict['K'], pose_init[:3])
                kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_triang_ids] - rep3d, axis=1)
                print(f'Triang points:{len(kps_rep_error)}'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}')

            kps2d_exist_ids = np.where(kpt2d3d_ids_f[start_idx:] < len(self.kpt3d_list))[0] + start_idx
            if len(kps2d_exist_ids) > 0:
                kps2d_nonzero_ids = np.where(kpt2d3d_ids_f[start_idx:] >= 0)[0] + start_idx
                kps2d_exist_ids = np.intersect1d(kps2d_exist_ids, kps2d_nonzero_ids)
                if len(kps2d_exist_ids) != 0:
                    kpt3d_exists_id = kpt2d3d_ids_f[kps2d_exist_ids]
                    kpt3d_exist = kpt3d_list_f[kpt3d_exists_id]
                    # rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
                    rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], pose_init[:3])
                    kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_exist_ids] - rep3d_exist, axis=1)
                    print(f'Exist points: - {len(kps_rep_error)}\n'
                          f'- min:{np.min(kps_rep_error)}\n'
                          f'- max:{np.max(kps_rep_error)}\n'
                          f'- med:{np.median(kps_rep_error)}')
        self.tm.tock('7-update triangulation')

        self.tm.tick('8-BA')
        # Apply BA with deep LM
        kpt3d_list_f, cams_f = self.apply_ba(kpt2ds_f, kpt2d3d_ids_f, kpt2d_fids_f,
                                                kpt3d_list_f, cams_f, verbose=False)

        K_opt, pose_opt = self.get_cam_params_back(cams_f[-1])

        trans_dist_pred, rot_dist_pred = self.cm_degree_5_metric(frame_info_dict['pose_pred'],
                                                                 frame_info_dict['pose_gt'])
        trans_dist_pred = np.round(trans_dist_pred, decimals=2)
        rot_dist_pred = np.round(rot_dist_pred, decimals=2)
        ba_log['pred_err_trans'] = trans_dist_pred
        ba_log['pred_err_rot'] = rot_dist_pred
        print(f"Pred pose error:{ba_log['pred_err_trans']} - {ba_log['pred_err_rot']}")

        trans_dist_init, rot_dist_init = self.cm_degree_5_metric(pose_init, frame_info_dict['pose_gt'])
        trans_dist_init = np.round(trans_dist_init, decimals=2)
        rot_dist_init = np.round(rot_dist_init, decimals=2)
        ba_log['init_err_trans'] = trans_dist_init
        ba_log['init_err_rot'] = rot_dist_init
        print(f"Initial pose error:{ba_log['init_err_trans']} - {ba_log['init_err_rot']}")

        trans_dist, rot_dist = self.cm_degree_5_metric(pose_opt, frame_info_dict['pose_gt'])
        trans_dist = np.round(trans_dist, decimals=2)
        rot_dist = np.round(rot_dist, decimals=2)
        ba_log['opt_err_trans'] = trans_dist
        ba_log['opt_err_rot'] = rot_dist
        print(f"Optimized pose error:{ba_log['opt_err_trans']} - {ba_log['opt_err_rot']}")
        self.tm.tock('8-BA')

        update_valid = True

        if (self.frame_id % self.frame_interval == 0 and update_valid) or self.init_cnt > 0:
            self.tm.tick('9-update')
            print(f"Num updated :{len(triang_keep_idx)}")

            unmatched_idx = np.array([i for i in range(len(kpt2ds_pred_query['keypoints'])) if i not in match_kq])
            num_unmatch = len(unmatched_idx)
            self.kpt2ds = np.concatenate([kpt2ds_f, kpt2ds_pred_query['keypoints'][unmatched_idx]])
            self.kpt2d_descs = np.concatenate([self.kpt2d_descs,
                                               kpt2ds_pred_query['descriptors'][:, match_kq[valid]].transpose(),
                                               kpt2ds_pred_query['descriptors'][:, unmatched_idx].transpose()])

            self.kpt2d_fids = np.concatenate([kpt2d_fids_f, np.ones(num_unmatch, dtype=int) * kpt2d_fids_f[-1]])
            self.cams = cams_f
            # self.kpt2ds_match = kpt2ds_match_f
            self.kpt3d_source_id = np.concatenate(
                [self.kpt3d_source_id, np.ones([len(kpt3d_list_f) - len(self.kpt3d_list)],
                                               dtype=int) * self.id])
            self.kpt3d_list = kpt3d_list_f
            self.kpt2d3d_ids = np.concatenate([kpt2d3d_ids_f, np.ones(num_unmatch, dtype=int) * -1])
            frame_info_dict['pose_pred'] = pose_init

            self.kf_kpt_index_dict[self.id] = (len(self.kpt2d_fids) - 1 - len(kpt2ds_pred_query['keypoints']),
                                               len(self.kpt2d_fids) - 1)
            self.kf_frames.pop(self.last_kf_id)
            self.kf_frames[self.id] = frame_info_dict

            self.last_kf_id = self.id
            self.id += 1

            if np.max(self.kpt2d_fids) > self.win_size:
                print("[START Deleting frame infos]")

                valid_idx_fid = np.where(self.kpt2d_fids > np.max(self.kpt2d_fids) - self.win_size)
                num_del = len(self.kpt2d_fids) - len(valid_idx_fid)
                start_idx, end_idx = self.kf_kpt_index_dict[self.last_kf_id]
                self.kf_kpt_index_dict[self.last_kf_id] = [start_idx - num_del, end_idx - num_del]
                self.kpt2d_fids = self.kpt2d_fids[valid_idx_fid]
                self.kpt2ds = self.kpt2ds[valid_idx_fid]
                self.kpt2d3d_ids = self.kpt2d3d_ids[valid_idx_fid]
                self.kpt2d_descs = self.kpt2d_descs[valid_idx_fid]
            self.tm.tock('9-update')

        self.frame_id += 1
        return pose_opt, ba_log, update_valid

    def track(self, frame_info_dict, flow_track_only=False, auto_mode=False):
        self.tm.tick('0-pose_init')
        if not auto_mode:
            # self.init_cnt = -1

            pose_ftk, kpt_dist = self.flow_track(frame_info_dict, self.last_kf_info)

            trans_dist_fkt, rot_dist_fkt = self.cm_degree_5_metric(self.pose_list[-1], pose_ftk)

            if len(self.pose_list) < 3:
                pose_mo = frame_info_dict['pose_pred']
            else:
                pose_mo = self.motion_prediction()

            # [Option] use motion prediction when keypoint tracking is not valid
            if ((trans_dist_fkt > 10 or rot_dist_fkt > 10) and self.use_motion_cnt < 3) or kpt_dist > 10:
                frame_info_dict['pose_init'] = pose_mo
                self.use_motion_cnt += 1
            elif (trans_dist_fkt < 20 and rot_dist_fkt < 20):
                frame_info_dict['pose_init'] = pose_ftk
                self.use_motion_cnt = 0
            else:
                frame_info_dict['pose_init'] = pose_mo
                self.use_motion_cnt += 1

            # frame_info_dict['pose_init'] = pose_mo

            # initialize with gt
            if self.init_cnt > 0:
                print("======= INITIALIZING")
                self.init_cnt -= 1
                frame_info_dict['pose_init'] = frame_info_dict['pose_gt']
        else:
            if self.init_cnt > 0:
                print("======= INITIALIZING")
                self.init_cnt -= 1
                frame_info_dict['pose_init'] = frame_info_dict['pose_gt']
            else:
                pose_mo = self.motion_prediction()
                frame_info_dict['pose_init'] = pose_mo
        self.tm.tock('0-pose_init')

        if not flow_track_only:
            self.tm.tick('0-track_ba')
            pose_opt, ba_log, update_valid = self.track_ba(frame_info_dict, verbose=False)
            self.pose_list.append(pose_opt)
            self.tm.tock('0-track_ba')
        else:
            pose_init = frame_info_dict['pose_init']
            pose_opt = frame_info_dict['pose_init']
            ba_log = dict()
            trans_dist_pred, rot_dist_pred = self.cm_degree_5_metric(frame_info_dict['pose_pred'],
                                                                     frame_info_dict['pose_gt'])
            trans_dist_pred = np.round(trans_dist_pred, decimals=2)
            rot_dist_pred = np.round(rot_dist_pred, decimals=2)
            ba_log['pred_err_trans'] = trans_dist_pred
            ba_log['pred_err_rot'] = rot_dist_pred
            print(f"Pred pose error:{ba_log['pred_err_trans']} - {ba_log['pred_err_rot']}")

            trans_dist_init, rot_dist_init = self.cm_degree_5_metric(pose_init, frame_info_dict['pose_gt'])
            trans_dist_init = np.round(trans_dist_init, decimals=2)
            rot_dist_init = np.round(rot_dist_init, decimals=2)
            ba_log['init_err_trans'] = trans_dist_init
            ba_log['init_err_rot'] = rot_dist_init
            print(f"Initial pose error:{ba_log['init_err_trans']} - {ba_log['init_err_rot']}")

            trans_dist, rot_dist = self.cm_degree_5_metric(pose_opt, frame_info_dict['pose_gt'])
            trans_dist = np.round(trans_dist, decimals=2)
            rot_dist = np.round(rot_dist, decimals=2)
            ba_log['opt_err_trans'] = trans_dist
            ba_log['opt_err_rot'] = rot_dist
            print(f"Optimized pose error:{ba_log['opt_err_trans']} - {ba_log['opt_err_rot']}")
            self.pose_list.append(frame_info_dict['pose_init'])
        print(self.tm.report())

        return frame_info_dict['pose_init'], pose_opt, ba_log
