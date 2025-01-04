import json
import os
import numpy as np
from torch.utils import data
import time 
import torch

""" Adapted by https://github.com/valeoai/rangevit """
class Nuscenes(data.Dataset):
    def __init__(self, dataroot, version='v1.0-mini', split='train', info_file_path="data_managment/nuscenes_info.json"):
        #assert version in ['v1.0-trainval', 'v1.0-mini']
        assert split in ['train', 'val', 'test']
        self.version = version
        self.split = split
        self.dataroot = dataroot
        
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        self.lidar_filenames = info_data[version][split]

    def __len__(self):
        return len(self.lidar_filenames)

    def loadDataByIndex(self, index):
        lidar_path = os.path.join(self.dataroot, self.lidar_filenames[index])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
        pointcloud = raw_data[:, :4]
        
        return pointcloud