import json
import os
import numpy as np
from torch.utils import data
import time 
import torch

CHANNELS = {'all':[2,3,4,5,6,7],
            'avia':[5,6,7],
            'horizon':[2,3,4]}

class FrontCam(data.Dataset):
    def __init__(self, dataroot='/data', version='v1.0-trainval', split='train', info_file_path="data_managment/frontcam.json", channels="all"):
        assert version in ['v1.0-trainval','v1.0-mini','v1.0-batch','v1.0-test']
        assert split in ['train', 'val', 'test']
        self.version = version
        self.split = split
        self.dataroot = dataroot
        self.channels = CHANNELS[channels]
        
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        self.lidar_filenames = info_data[version][split]

    def __len__(self):
        return len(self.lidar_filenames)

    def loadDataByIndex(self, index):
        lidar_path = os.path.join(self.dataroot, self.lidar_filenames[index])
        raw_data = np.load(lidar_path)['point_cloud']
        timestamp = np.load(lidar_path)['timestamp']

        all_points = []
        for i,channel in enumerate(self.channels):
            data_channel = raw_data[np.where(raw_data[:,4] == channel)]

            pcd = data_channel[:,:3]
            all_points.append(pcd)

        pointcloud = np.vstack(all_points)
        depth = np.linalg.norm(pointcloud,axis=1)
        pitch = np.arcsin(pointcloud[:,2] / depth)

        mask = np.logical_or(pitch < -5*(np.pi/180), np.logical_and(depth<3.7, np.abs(pointcloud[:,1])<0.9))
        
        pointcloud = pointcloud[~mask,:]
        
        return pointcloud