import cv2
try:
    import ujson as json
except ImportError:
    import json
import torch
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from src.utils import data_utils


class GATsSPGDataset(Dataset):
    def __init__(
        self, 
        anno_file, 
        num_leaf, 
        split,
        pad=True, 
        shape2d=1000, 
        shape3d=2000, 
        pad_val=0,
        load_pose_gt=False,
    ):
        super(Dataset, self).__init__()
        
        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())
        self.num_leaf = num_leaf

        self.split = split
        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d
        self.pad_val = pad_val
        self.load_pose_gt = load_pose_gt

    def get_intrin_by_color_path(self, color_path):
        intrin_path = color_path.replace('/color/', '/intrin_ba/').replace(
            '.png', '.txt'
        )
        K_crop = torch.from_numpy(np.loadtxt(intrin_path)) # [3, 3]
        return K_crop
        
    def get_gt_pose_by_color_path(self, color_path):
        gt_pose_path = color_path.replace('/color/', '/poses_ba/').replace(
            '.png', '.txt'
        )
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path)) # [4, 4]
        return pose_gt
    
    def read_anno2d(self, anno2d_file, height, width, pad=True):
        """ Read (and pad) 2d info"""
        with open(anno2d_file, 'r') as f:
            data = json.load(f)
        
        keypoints2d = torch.Tensor(data['keypoints2d']) # [n, 2]
        descriptors2d = torch.Tensor(data['descriptors2d']) # [dim, n]
        scores2d = torch.Tensor(data['scores2d']) # [n, 1]
        assign_matrix = torch.Tensor(data['assign_matrix']) # [2, k]

        num_2d_orig = keypoints2d.shape[0]

        if pad:
            keypoints2d, descriptors2d, scores2d = data_utils.pad_keypoints2d_random(keypoints2d, descriptors2d, scores2d,
                                                                                     height, width, self.shape2d)
        return keypoints2d, descriptors2d, scores2d, assign_matrix, num_2d_orig
    
    def read_anno3d(self, avg_anno3d_file, clt_anno3d_file, idxs_file, pad=True):
        """ Read(and pad) 3d info"""
        avg_data = np.load(avg_anno3d_file)
        clt_data = np.load(clt_anno3d_file)
        idxs = np.load(idxs_file)

        keypoints3d = torch.Tensor(clt_data['keypoints3d']) # [m, 3]
        avg_descriptors3d = torch.Tensor(avg_data['descriptors3d']) # [dim, m]
        clt_descriptors = torch.Tensor(clt_data['descriptors3d']) # [dim, k]
        avg_scores = torch.Tensor(avg_data['scores3d']) # [m, 1]
        clt_scores = torch.Tensor(clt_data['scores3d'])  # [k, 1]

        num_3d_orig = keypoints3d.shape[0]
        if pad:
            keypoints3d = data_utils.pad_keypoints3d_random(keypoints3d, self.shape3d)
            avg_descriptors3d, avg_scores = data_utils.pad_features3d_random(avg_descriptors3d, avg_scores, self.shape3d)
            clt_descriptors, clt_scores = data_utils.build_features3d_leaves(clt_descriptors, clt_scores, idxs,
                                                                            self.shape3d, num_leaf=self.num_leaf)
        return keypoints3d, avg_descriptors3d, avg_scores, clt_descriptors, clt_scores, num_3d_orig
    
    def read_anno(self, img_id):
        """
        Read image, 2d info and 3d info.
        Pad 2d info and 3d info to a constant size.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        color_path = self.coco.loadImgs(int(img_id))[0]['img_file']
        image = cv2.imread(color_path)
        height, width, _ = image.shape

        idxs_file = anno['idxs_file']
        avg_anno3d_file = anno['avg_anno3d_file']
        collect_anno3d_file = anno['collect_anno3d_file']

        # Load 3D points and features:
        (
            keypoints3d, 
            avg_descriptors3d, 
            avg_scores3d, 
            clt_descriptors2d, 
            clt_scores2d, 
            num_3d_orig 
        ) = self.read_anno3d(avg_anno3d_file, collect_anno3d_file, idxs_file, pad=self.pad)
        
        if self.split == 'train': 
            anno2d_file = anno['anno2d_file']
            # Load 2D keypoints, features and GT 2D-3D correspondences:
            (
                keypoints2d, 
                descriptors2d, 
                scores2d, 
                assign_matrix, 
                num_2d_orig
            ) = self.read_anno2d(anno2d_file, height, width, pad=self.pad)
            
            # Construct GT conf_matrix:
            conf_matrix = data_utils.reshape_assign_matrix(
                            assign_matrix, 
                            num_2d_orig, 
                            num_3d_orig,
                            self.shape2d, 
                            self.shape3d, 
                            pad=True, 
                            pad_val=self.pad_val
                        )

            data = {
                'keypoints2d': keypoints2d, # [n1, 2]
                'descriptors2d_query': descriptors2d, # [dim, n1]
            }

        elif self.split == 'val':
            image_gray = data_utils.read_gray_scale(color_path) / 255.
            data = {
                'image': image_gray
            }
            conf_matrix = torch.Tensor([])

        data.update({
            'keypoints3d': keypoints3d, # [n2, 3]
            'descriptors3d_db': avg_descriptors3d, # [dim, n2]
            'descriptors2d_db': clt_descriptors2d, # [dim, n2 * num_leaf]
            'image_size': torch.Tensor([height, width])
        })
        
        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_path(color_path)
            pose_gt = self.get_gt_pose_by_color_path(color_path)
            data.update({'query_intrinsic': K_crop, 'query_pose_gt': pose_gt, 'query_image': image})

        return data, conf_matrix
    
    def __getitem__(self, index):
        img_id = self.anns[index]

        data, conf_matrix = self.read_anno(img_id)
        return data, conf_matrix
    
    def __len__(self):
        return len(self.anns)