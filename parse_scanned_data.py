import os
import cv2
import tqdm
import numpy as np
import os.path as osp
import argparse
from pathlib import Path
from transforms3d import affines, quaternions
from src.utils import data_utils

def get_arkit_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')

    color_dir = osp.join(data_dir, 'color')
    Path(color_dir).mkdir(parents=True, exist_ok=True)

    box_file = osp.join(data_dir, 'Box.txt')
    assert Path(box_file).exists()
    out_3D_box_dir = osp.join(osp.dirname(data_dir), 'box3d_corners.txt')

    out_pose_dir = osp.join(data_dir, 'poses')
    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    pose_file = osp.join(data_dir, 'ARposes.txt')
    assert Path(pose_file).exists()

    reproj_box_dir = osp.join(data_dir, 'reproj_box')
    Path(reproj_box_dir).mkdir(parents=True, exist_ok=True)
    out_box_dir = osp.join(data_dir, 'bbox')
    Path(out_box_dir).mkdir(parents=True, exist_ok=True)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    assert Path(orig_intrin_file).exists()

    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    intrin_dir = osp.join(data_dir, 'intrin')
    Path(intrin_dir).mkdir(parents=True, exist_ok=True)

    M_dir = osp.join(data_dir, 'M')
    Path(M_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'video_file': video_file,
        'color_dir': color_dir,
        'box_path': box_file,
        'pose_file': pose_file,
        'out_box_dir': out_box_dir,
        'out_3D_box_dir': out_3D_box_dir,
        'reproj_box_dir': reproj_box_dir,
        'out_pose_dir': out_pose_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
        'intrin_dir': intrin_dir,
        'M_dir': M_dir
    }

    return paths


def get_test_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')

    # box_file = osp.join(data_dir, 'RefinedBox.txt')
    box_file = osp.join(data_dir, 'Box.txt')
    if osp.exists(box_file):
        os.remove(box_file)

    color_full_dir = osp.join(data_dir, 'color_full')
    Path(color_full_dir).mkdir(parents=True, exist_ok=True)

    pose_file = osp.join(data_dir, 'ARposes.txt')
    if osp.exists(pose_file):
        os.remove(pose_file)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    paths = {
        'video_file': video_file,
        'color_full_dir': color_full_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
    }

    return paths


def get_bbox3d(box_path):
    assert Path(box_path).exists()
    with open(box_path, 'r') as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(',')]
    ex, ey, ez = box_data[3: 6]
    bbox_3d = np.array([
        [-ex, -ey, -ez],
        [ex,  -ey, -ez],
        [ex,  -ey, ez],
        [-ex, -ey, ez],
        [-ex,  ey, -ez],
        [ ex,  ey, -ez],
        [ ex,  ey, ez],
        [-ex,  ey, ez]
    ]) * 0.5
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    return bbox_3d, bbox_3d_homo

def parse_box(box_path):
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    quaternion = data[6:]
    rot_mat = quaternions.quat2mat(quaternion)
    T_ow = affines.compose(position, rot_mat, np.ones(3))
    return T_ow


def reproj(K_homo, pose, points3d_homo):
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4 # [4 ,n]

    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:]  / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]


def parse_video(paths, downsample_rate=5, bbox_3d_homo=None, hw=512):
    orig_intrin_file = paths['final_intrin_file']
    K, K_homo = data_utils.get_K(orig_intrin_file)

    intrin_dir = paths['intrin_dir']
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if index % downsample_rate == 0:
            img_name = osp.join(paths['color_dir'], '{}.png'.format(index))
            save_intrin_path = osp.join(intrin_dir, '{}.txt'.format(index))

            reproj_box3d_file = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
            if not osp.isfile(reproj_box3d_file):
                continue
            reproj_box3d = np.loadtxt(osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))).astype(int)
            x0, y0 = reproj_box3d.min(0)
            x1, y1 = reproj_box3d.max(0)

            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
            image_crop, trans1 = data_utils.get_image_crop_resize(image, box, resize_shape)

            box_new = np.array([0, 0, x1-x0, y1-y0])
            resize_shape = np.array([hw, hw])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop, trans2 = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)

            trans_full_to_crop = trans2 @ trans1
            trans_crop_to_full = np.linalg.inv(trans_full_to_crop)

            np.savetxt(osp.join(paths['M_dir'], '{}.txt'.format(index)), trans_crop_to_full)

            pose = np.loadtxt(osp.join(paths['out_pose_dir'], '{}.txt'.format(index)))
            reproj_crop = reproj(K_crop_homo, pose, bbox_3d_homo.T)
            x0_new, y0_new = reproj_crop.min(0)
            x1_new, y1_new = reproj_crop.max(0)
            box_new = np.array([x0_new, y0_new, x1_new, y1_new])

            np.savetxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index)), box_new)
            cv2.imwrite(img_name, image_crop)
            # cv2.imwrite(out_mask_file, mask_crop)
            full_img_dir = paths['color_dir'] + '_full'
            Path(full_img_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
            np.savetxt(save_intrin_path, K_crop)

        index += 1
    cap.release()


def data_process_anno(data_dir, downsample_rate=1, hw=512):
    paths = get_arkit_default_path(data_dir)
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))

    bbox_3d, bbox_3d_homo = get_bbox3d(paths['box_path'])
    np.savetxt(paths['out_3D_box_dir'], bbox_3d)

    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    with open(paths['pose_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        index = 0
        for line in tqdm.tqdm(lines):
            if len(line) == 0 or line[0] == '#':
                continue

            if index % downsample_rate == 0:
                eles = line.split(',')
                data = [float(e) for e in eles]

                position = data[1:4]
                quaternion = data[4:]
                rot_mat = quaternions.quat2mat(quaternion)
                rot_mat = rot_mat @ np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])

                T_ow = parse_box(paths['box_path'])
                T_cw = affines.compose(position, rot_mat, np.ones(3))
                T_wc = np.linalg.inv(T_cw)
                T_oc = T_wc @ T_ow
                pose_save_path = osp.join(paths['out_pose_dir'], '{}.txt'.format(index))
                box_save_path = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
                reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

                x0, y0 = reproj_box3d.min(0)
                x1, y1 = reproj_box3d.max(0)

                if x0 < -1000 or y0 < -1000 or x1 > 3000 or y1 > 3000:
                    continue

                np.savetxt(pose_save_path, T_oc)
                np.savetxt(box_save_path, reproj_box3d)
            index += 1

    parse_video(paths, downsample_rate, bbox_3d_homo, hw=hw)

    # Make fake data for demo annotate video without BA:
    if osp.exists(osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba')):
        os.system(f"rm -rf {osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba')}")
    os.system(f"ln -s {paths['intrin_dir']} {osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba')}")

    if osp.exists(osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba')):
        os.system(f"rm -rf {osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba')}")
    os.system(f"ln -s {paths['out_pose_dir']} {osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba')}")

def data_process_test(data_dir, downsample_rate=1):
    paths = get_test_default_path(data_dir)

    # Parse intrinsic:
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))
    
    # Parse video:
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if index % downsample_rate == 0:
            full_img_dir = paths['color_full_dir']
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
        index += 1
    cap.release()

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--scanned_object_path", type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.scanned_object_path
    assert osp.exists(data_dir), f"Scanned object path:{data_dir} not exists!"

    seq_dirs = os.listdir(data_dir)
    for seq_dir in seq_dirs:
        if '-annotate' in seq_dir:
            print('=> Processing annotate sequence: ', seq_dir)
            data_process_anno(osp.join(data_dir, seq_dir), downsample_rate=1, hw=512)
        elif '-test' in seq_dir:
            # Parse scanned test sequence
            print('=> Processing test sequence: ', seq_dir)
            data_process_test(osp.join(data_dir, seq_dir), downsample_rate=1)
        else:
            continue

