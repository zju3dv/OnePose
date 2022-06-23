import numpy as np
import torch


class Timer(object):
    def __init__(self):
        self.time_dict = dict()
        self.res_dict = dict()
        self.stash_dict = dict()

    def set(self, label, value):
        if label not in self.stash_dict.keys():
            self.stash_dict[label] = []
        self.stash_dict[label].append(value)

    def tick(self, tick_name):
        import time
        self.time_dict[tick_name] = time.time() * 1000.0
        return self.time_dict[tick_name]

    def tock(self, tock_name, pop=False):
        if tock_name not in self.time_dict:
            return 0.0
        else:
            import time
            t2 = time.time() * 1000.0
            t1 = self.time_dict[tock_name]
            self.res_dict[tock_name] = t2 - t1
            if pop:
                self.time_dict.pop(tock_name)
            return self.res_dict[tock_name]

    def stash(self):
        for k, v in self.res_dict.items():
            if k not in self.stash_dict.keys():
                self.stash_dict[k] = []
            self.stash_dict[k].append(v)

    def report_stash(self):
        res_dict = dict()
        for k, v in self.stash_dict.items():
            res_dict[k] = np.mean(v)
        return res_dict

    def report(self):
        return self.res_dict


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    def to_homogeneous(points):
        return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def project(xyz, K, RT, need_depth=False):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T)
    xyz += RT[:, 3:].T
    depth = xyz[:, 2:].flatten()
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    if need_depth:
        return xy, depth
    else:
        return xy


def AngleAxisRotatePoint(angleAxis, pt):
    theta2 = (angleAxis * angleAxis).sum(dim=1)

    mask = (theta2 > 0).float()

    theta = torch.sqrt(theta2 + (1 - mask))

    mask = mask.reshape((mask.shape[0], 1))
    mask = torch.cat([mask, mask, mask], dim=1)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    thetaInverse = 1.0 / theta

    w0 = angleAxis[:, 0] * thetaInverse
    w1 = angleAxis[:, 1] * thetaInverse
    w2 = angleAxis[:, 2] * thetaInverse

    wCrossPt0 = w1 * pt[:, 2] - w2 * pt[:, 1]
    wCrossPt1 = w2 * pt[:, 0] - w0 * pt[:, 2]
    wCrossPt2 = w0 * pt[:, 1] - w1 * pt[:, 0]

    tmp = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp

    r0 = r0.reshape((r0.shape[0], 1))
    r1 = r1.reshape((r1.shape[0], 1))
    r2 = r2.reshape((r2.shape[0], 1))

    res1 = torch.cat([r0, r1, r2], dim=1)

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    r00 = r00.reshape((r00.shape[0], 1))
    r01 = r01.reshape((r01.shape[0], 1))
    r02 = r02.reshape((r02.shape[0], 1))

    res2 = torch.cat([r00, r01, r02], dim=1)

    return res1 * mask + res2 * (1 - mask)


def SnavelyReprojectionErrorV2(points_ob, cameras_ob, features):
	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]
	focals = features[:, 2]
	l1 = features[:, 3]
	l2 = features[:, 4]

    # camera[0,1,2] are the angle-axis rotation.
	p = AngleAxisRotatePoint(cameras_ob[:, :3], points_ob)
	p = p + cameras_ob[:, 3:6]

	xp = p[:,0] / p[:,2]
	yp = p[:,1] / p[:,2]

	# predicted_x, predicted_y = DistortV2(xp, yp, cameras_ob, cam_K)

	predicted_x = focals * xp + l1
	predicted_y = focals * yp + l2

	residual_0 = predicted_x - features[:, 0]
	residual_1 = predicted_y - features[:, 1]

	residual_0 = residual_0.reshape((residual_0.shape[0], 1))
	residual_1 = residual_1.reshape((residual_1.shape[0], 1))

	#return torch.sqrt(residual_0**2 + residual_1 ** 2)
	return torch.cat([residual_0, residual_1], dim=1)


def put_text(img, inform_text, color=None):
    import cv2
    fontScale = 1
    if color is None:
        color = (255, 0, 0)
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    img = cv2.putText(img, inform_text, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img


def draw_kpt2d(image, kpt2d, color=(0, 0, 255), radius=2, thikness=1):
    import cv2
    for coord in kpt2d:
        cv2.circle(image, (int(coord[0]), int(coord[1])), radius, color, thikness, 1)
        # cv2.circle(image, (int(coord[0]), int(coord[1])), 7, color, 1, 1)
    return image


class MovieWriter:
    def __init__(self):
        self.video_out_path = ''
        self.movie_cap = None
        self.id = 0

    def start(self):
        if self.movie_cap is not None:
            self.movie_cap.release()

    def write(self, im_bgr, video_out_path, text_info=[], fps=20):
        import cv2
        if self.movie_cap is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            length = fps
            self.video_out_path = video_out_path
            self.movie_cap = cv2.VideoWriter(video_out_path, fourcc,
                                             length, (im_bgr.shape[1], im_bgr.shape[0]))

        if len(text_info) > 0:
            self.put_text(im_bgr, text_info[0], color=text_info[1])
        self.movie_cap.write(im_bgr)
        self.id += 1

    def put_text(self, img, inform_text, color=None):
        import cv2
        fontScale = 1
        if color is None:
            color = (255, 0, 0)
        org = (200, 200)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        img = cv2.putText(img, inform_text, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return img

    def end(self):
        if self.movie_cap is not None:
            self.movie_cap.release()
            self.movie_cap = None
            print(f"Output frames:{self.id} to {self.video_out_path}")
            self.id = 0
