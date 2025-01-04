import json
import os
import numpy as np
from torch.utils import data



class Kitti(data.Dataset):
    def __init__(self, split='train', info_file_path="data_managment/nuscenes_info.json"):
        assert split in ['train', 'val', 'test']
        self.split = split
        
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        self.lidar_filenames = info_data[split]

    def __len__(self):
        return len(self.lidar_filenames)

    def loadDataByIndex(self, index):
        lidar_path = self.lidar_filenames[index]
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
        pointcloud = raw_data[:, :4]
        
        return pointcloud