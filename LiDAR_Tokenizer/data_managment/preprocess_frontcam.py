import json
import os
import numpy as np
from torch.utils import data
import time 
import torch

from .projections import pcd2range, process_scan
from LiDAR_Tokenizer.model_managment.helpers import instantiate_from_config

import yaml

""" This is just pieces of code extracted from the torch lightning dataset classes,
so that it can be used standalone without using that pipeline (specifically it is used 
in MaskGIT pretokenization). """

def preprocess_range(fns, config_path, channels=[2,3,4,5,6,7]):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = config['model']['params']['lossconfig']['params']['dataset_config']

    sensor_pos = config['sensor_pos']
    proj_h = config['size'][0]
    proj_w = config['size'][1]
    fov = config['fov']
    depth_range = config['depth_range']
    t_ratio = config['t_ratio']
    b_ratio = config['b_ratio']
    l_ratio = config['l_ratio']
    r_ratio = config['r_ratio']
    log_scale = config['log_scale']
    depth_scale = config['depth_scale']
    if log_scale:
        depth_thresh = (np.log2(1./255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        depth_thresh = (1./255. / depth_scale) * 2. - 1 + 1e-6

    range_images = []
    for fn in fns:
        raw_data = np.load(fn)['point_cloud']
        timestamp = np.load(fn)['timestamp']

        all_points = []
        for i,channel in enumerate(channels):
            data_channel = raw_data[np.where(raw_data[:,4] == channel)]

            pcd = data_channel[:,:3]
            all_points.append(pcd)

        pointcloud = np.vstack(all_points)
        depth = np.linalg.norm(pointcloud,axis=1)
        pitch = np.arcsin(pointcloud[:,2] / depth)

        mask = np.logical_or(pitch < -5*(np.pi/180), np.logical_and(depth<3.7, np.abs(pointcloud[:,1])<0.9))
        
        pointcloud = pointcloud[~mask,:]

        pointcloud_xyz = pointcloud[:, :3]
        pointcloud_xyz -= sensor_pos

        range_img = pcd2range(pointcloud_xyz, [proj_h, proj_w], fov, depth_range)

        t = int(round(t_ratio*proj_h))
        b = int(round(b_ratio*proj_h))
        l = int(round(l_ratio*proj_w))
        r = int(round(r_ratio*proj_w))

        range_img = range_img[t:b, l:r]

        proj_range, proj_mask = process_scan(range_img, log_scale, depth_scale, depth_thresh)

        range_images.append(proj_range)
    range_images = np.stack(range_images)
    return range_images


def preprocess_voxel(fns, config_path, channels=[2,3,4,5,6,7]):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    voxelizer = instantiate_from_config(config['model']['params']['voxelizer'])
    config = config['model']['params']['voxelizer']['params']

    all_voxels = []
    for fn in fns:
        raw_data = np.load(fn)['point_cloud']
        timestamp = np.load(fn)['timestamp']

        all_points = []
        for i,channel in enumerate(channels):
            data_channel = raw_data[np.where(raw_data[:,4] == channel)]

            pcd = data_channel[:,:3]
            all_points.append(pcd)

        pointcloud = np.vstack(all_points)
        depth = np.linalg.norm(pointcloud,axis=1)
        pitch = np.arcsin(pointcloud[:,2] / depth)

        mask = np.logical_or(pitch < -5*(np.pi/180), np.logical_and(depth<3.7, np.abs(pointcloud[:,1])<0.9))
        
        pointcloud = pointcloud[~mask,:]

        xyz = pointcloud[:, :3]

        xyz = xyz[np.where((xyz[:,2] > config['z_min']) & (xyz[:,2] < config['z_max']))[0],:]
        # log voxelizer assumption
        xyz = xyz[np.where((xyz[:,0] > config['x_min']) & (xyz[:,0] < config['x_max']))[0],:]  
        max_angle = config['fov']/2
        xyz = xyz[np.where(np.arctan2(np.abs(xyz[:,1]), np.abs(xyz[:,0])) < max_angle )[0],:]

        voxels = voxelizer([[torch.from_numpy(xyz)]])
        all_voxels.append(voxels)
    all_voxels = torch.stack(all_voxels)
    return all_voxels

