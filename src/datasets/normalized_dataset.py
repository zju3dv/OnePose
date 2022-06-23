import cv2
import numpy as np

from types import SimpleNamespace
from torch.utils.data import Dataset


class NormalizedDataset(Dataset):
    """read images(suppose images have been cropped)"""
    default_conf = {
        'globs': ['*.jpg', '*.png'],
        'grayscale': True,
    }

    def __init__(self, img_lists, conf):
        self.img_lists = img_lists
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        
        if len(img_lists) == 0:
            raise ValueError('Could not find any image.')
    
    def __getitem__(self, index):
        img_path = self.img_lists[index]

        mode = cv2.IMREAD_GRAYSCALE if self.conf.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(img_path, mode)
        size = image.shape[:2]

        image = image.astype(np.float32)
        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))
        image /= 255.

        data = {
            'path': str(img_path),
            'image': image,
            'size': np.array(size),    
        }
        return data
    
    def __len__(self):
        return len(self.img_lists)