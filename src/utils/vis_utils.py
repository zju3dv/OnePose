import cv2
import os
from pathlib import Path
from PIL import Image
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import natsort
from loguru import logger

matplotlib.use("Agg")
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]


def plot_image_pair(imgs, dpi=100, size=6, pad=0.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two.'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values(): # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, marker='x')
    ax[1].scatter(kpts1[:, 1], kpts1[:, 1], c=color, s=ps, marker='x')


def plot_matches(kpts0, kpts1, color, lw=0.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigures = fig.transFigure.inverted()
    fkpts0 = transFigures.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigures.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    fast_viz=False,
    opencv_display=False,
    opencv_title='matches',
    small_text=[],
):
    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints
        )
        return 

    plot_image_pair([image0, image1]) # will create a new figure
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :100].mean() > 200 else 'w'
    fig.text(
        0.01,
        0.99,
        '\n'.join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va='top',
        ha='left',
        color=txt_color,
    )

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01,
        0.01,
        '\n'.join(small_text),
        transform=fig.axes[0].transAxes,
        fontsize=5,
        va='bottom',
        ha='left',
        color=txt_color
    )
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def make_matching_plot_fast(image0, image1, kpts0, kpts1,
                            mkpts0, mkpts1, color, text,
                            margin=10, show_keypoints=True, num_matches_to_show=None):
    """draw matches in two images"""
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    
    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out] * 3, -1)
    
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), 
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)
    
    scale = min(H / 640., 2.0) * 1.2 # scale
    Ht = int(30 * scale) # text height
    text_color_fg = (0, 225, 255)
    text_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*scale), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*scale, text_color_bg, 3, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*scale), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*scale, text_color_fg, 1, cv2.LINE_AA)
        
    cv2.namedWindow('vis', 0)
    cv2.resizeWindow('vis', 800, 800)
    cv2.imshow('vis', out)
    cv2.waitKey(0)


def vis_match_pairs(pred, feats0, feats1, name0, name1):
    """vis matches on two images"""
    import matplotlib.cm as cm

    image0_path = name0
    image1_path = name1
    
    image0 = cv2.imread(image0_path)
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    
    matches = pred['matches0'][0].detach().cpu().numpy()
    valid = matches > -1
    
    kpts0, kpts1 = feats0['keypoints'].__array__(), feats1['keypoints'].__array__()
    mkpts0, mkpts1 = kpts0[valid], kpts1[matches[valid]]
    
    conf = pred['matching_scores0'][0].detach().cpu().numpy()
    mconf = conf[valid]
    color = cm.jet(mconf)

    make_matching_plot_fast(
        image0, image1, kpts0, kpts1,
        mkpts0, mkpts1, color, text=[]
    )


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K
    
    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose
    
    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]


def draw_3d_box(image, corners_2d, linewidth=3, color='g'):
    """ Draw 3d box corners 
    @param corners_2d: [8, 2]
    """
    lines = np.array([
        [0, 1, 5, 4, 2, 3, 7, 6, 2, 2, 3, 7],
        [1, 5, 4, 0, 3, 0, 6, 5, 1, 6, 7, 4]
    ]).T

    colors = {
        'g': (0, 255, 0),
        'r': (0, 0, 255),
        'b': (255, 0, 0)
    }
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]
    
    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)


def draw_2d_box(image, corners_2d, linewidth=3):
    """ Draw 2d box corners
    @param corners_2d: [x_left, y_top, x_right, y_bottom]
    """
    x1, y1, x2, y2 = corners_2d.astype(int)
    box_pts = [
        [(x1, y1), (x1, y2)],
        [(x1, y2), (x2, y2)],
        [(x2, y2), (x2, y1)],
        [(x2, y1), (x1, y1)]
    ]

    for pts in box_pts:
        pt1, pt2 = pts
        cv2.line(image, pt1, pt2, (0, 0, 255), linewidth)


def draw_reprojection_pair(data, val_results, visual_color_type='conf'):
    query_image = data['query_image'][0].cpu().numpy()
    query_K = data['query_intrinsic'][0].cpu().numpy()
    query_pose_gt = data['query_pose_gt'][0].cpu().numpy()
    mconf = val_results['mconf']
    mkpts3d = val_results['mkpts3d']
    mkpts2d = val_results['mkpts2d']
    mkpts3d_reprojed = reproj(query_K, query_pose_gt, mkpts3d)
    
    figures = {'evaluation': []}
    text = [
        f'Num of matches: {mkpts3d_reprojed.shape[0]}',
    ]
    if visual_color_type == 'conf':
        if mkpts3d_reprojed.shape[0] != 0:
            mconf_max = np.max(mconf)    
            mconf_min = np.min(mconf)
            mconf_normalized = (mconf - mconf_min) / (
                mconf_max - mconf_min + 1e-4
            )
            color = jet(mconf_normalized)

            text += [
                f'Max conf: {mconf_max}',
                f'Min conf: {mconf_min}',
            ]
        else:
            color = np.array([])

    elif visual_color_type == 'epi_error':
        raise NotImplementedError

    else:
        raise NotImplementedError
    
    figure = make_matching_plot(
        query_image,
        query_image,
        mkpts2d,
        mkpts3d_reprojed,
        mkpts2d,
        mkpts3d_reprojed,
        color=color,
        text=text
    )
    figures['evaluation'].append(figure)

    return figures
        

def vis_reproj(image_full_path, poses, box3d_path, intrin_full_path,
               save_demo=False, demo_root=None, colors=['y', 'g']):
    """ 
    Draw 2d box reprojected by 3d box.
    Yellow for gt pose, and green for pred pose.
    """
    def parse_K(intrin_full_path):
        """ Read intrinsics"""
        with open(intrin_full_path, 'r') as f:
            lines = [line.rstrip('\n').split(':')[1] for line in f.readlines()]
        
        fx, fy, cx, cy = list(map(float, lines))

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        K_homo = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0,  0,  1, 0]
        ])
        return K, K_homo # [3, 3], [3, 4]

    box3d = np.loadtxt(box3d_path)
    K_full, _  = parse_K(intrin_full_path)

    assert osp.isfile(image_full_path), "Please parse full image from \"Frames.m4v\" first."
    image_full = cv2.imread(image_full_path)

    for pose, color in zip(poses, colors):
        # Draw pred 3d box
        if pose is not None:
            reproj_box_2d = reproj(K_full, pose, box3d)
            draw_3d_box(image_full, reproj_box_2d, color=color)

    if save_demo:
        img_idx = int(osp.basename(image_full_path).split('.')[0])
        obj_name = image_full_path.split('/')[-3]
        demo_dir = osp.join(demo_root, obj_name)
        Path(demo_dir).mkdir(exist_ok=True, parents=True)

        save_path = osp.join(demo_dir, '{:05d}.jpg'.format(img_idx))
        print(f'=> Saving image: {save_path}')
        cv2.imwrite(save_path, image_full)

    return image_full

def save_demo_image(pose_pred, K, image_path, box3d_path, draw_box=True, save_path=None):
    """ 
    Project 3D bbox by predicted pose and visualize
    """
    box3d = np.loadtxt(box3d_path)

    image_full = cv2.imread(image_path)

    if draw_box:
        reproj_box_2d = reproj(K, pose_pred, box3d)
        draw_3d_box(image_full, reproj_box_2d, color='b', linewidth=10)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(save_path, image_full)
    return image_full

def dump_wis3d(idx, cfg, data_dir, image0, image1, image_full,
               kpts2d, kpts2d_reproj, confidence, inliers):
    """ Visualize by wis3d """
    from wis3d import Wis3D

    seq_name = '_'.join(data_dir.split('/')[-2:])
    if cfg.suffix:
        seq_name += '_' + cfg.suffix
    wis3d = Wis3D(cfg.output.vis_dir, seq_name)
    wis3d.set_scene_id(idx)

    # property for wis3d
    reproj_distance = np.linalg.norm(kpts2d_reproj - kpts2d, axis=1)
    inliers_bool = np.zeros((kpts2d.shape[0], 1), dtype=np.bool)
    if inliers is not None:
        inliers_bool[inliers] = True
        num_inliers = len(inliers)
    else:
        num_inliers = 0

    wis3d.add_keypoint_correspondences(image0, image1, kpts2d, kpts2d_reproj,
                                       metrics={
                                           'mconf': confidence.tolist(),
                                           'reproj_distance': reproj_distance.tolist()
                                       },
                                       booleans={
                                           'inliers': inliers_bool.tolist()
                                       },
                                       meta={
                                           'num_inliers': num_inliers,
                                           'width': image0.size[0],
                                           'height': image0.size[1],
                                       },
                                       name='matches')
    image_full_pil = Image.fromarray(cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB))
    wis3d.add_image(image_full_pil, name='results')

def make_video(image_path, output_video_path):
    # Generate video:
    images = natsort.natsorted(os.listdir(image_path))
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    H, W, C = cv2.imread(osp.join(image_path, images[0])).shape
    if osp.exists(output_video_path):
        os.remove(output_video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, 24, (W, H))
    for id, image_name in enumerate(images):
        image = cv2.imread(osp.join(image_path, image_name))
        video.write(image)
    video.release()
    logger.info(f"Demo vido saved to: {output_video_path}")