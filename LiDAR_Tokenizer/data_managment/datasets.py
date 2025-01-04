import numpy as np
import torch
from torch.utils.data import Dataset
import time

#from data_loading import augmentor

try:
    from data_managment.projections import pcd2range, process_scan
    from data_managment.augmentation import Augmentor,AugmentParams
    from data_managment.voxelize import Voxelizer,LogVoxelizer
    from model_managment.helpers import instantiate_from_config
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")
    from ..data_managment.projections import pcd2range, process_scan
    from ..data_managment.augmentation import Augmentor,AugmentParams
    from ..data_managment.voxelize import Voxelizer,LogVoxelizer
    from ..model_managment.helpers import instantiate_from_config

class Raw(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        pointcloud = self.dataset.loadDataByIndex(index)
        xyz = pointcloud[:, :3]
        return xyz
    

class Voxels(Dataset):
    def __init__(self, dataset, config, is_train=True, return_original=False):
        self.dataset = dataset
        self.is_train = is_train
        self.return_original = return_original

        voxelizer_config = config['model']['params']['voxelizer']

        if 'mode' in voxelizer_config:
            self.mode =             voxelizer_config['mode']
        else:
            self.mode = "linear" # default
            
        self.zmin =             voxelizer_config['params']['z_min']
        self.zmax =             voxelizer_config['params']['z_max']
        self.fov_horizontal =   voxelizer_config['params']['fov']

        if self.mode =="linear":
            self.xmin =     voxelizer_config['params']['x_min']
            self.xmax =     voxelizer_config['params']['x_max']
            self.ymin =     voxelizer_config['params']['y_min']
            self.ymax =     voxelizer_config['params']['y_max']

        elif self.mode =="log":
            self.xmin =     voxelizer_config['params']['x_min']
            self.xmax =     voxelizer_config['params']['x_max']

        elif self.mode =="polar":
            self.r_min =            voxelizer_config['params']['r_min']
            self.r_max =            voxelizer_config['params']['r_max']

        else:
            raise NotImplementedError(f"Choose as mode 'linear' 'log' or 'polar' not {self.mode}")
        
        self.voxelizer = instantiate_from_config(voxelizer_config)

    def __getitem__(self, index):
        pointcloud = self.dataset.loadDataByIndex(index)
            
        xyz = pointcloud[:, :3]

        # Filter z for every voxelizer.
        xyz = xyz[np.where((xyz[:,2] > self.zmin) & (xyz[:,2] < self.zmax))[0],:]

        if self.mode =="linear":
            xyz = xyz[np.where((xyz[:,0] > self.xmin) & (xyz[:,0] < self.xmax))[0],:]  
            xyz = xyz[np.where((xyz[:,1] > self.ymin) & (xyz[:,1] < self.ymax))[0],:]

        elif self.mode =="log":
            xyz = xyz[np.where((xyz[:,0] > self.xmin) & (xyz[:,0] < self.xmax))[0],:]  
            max_angle = self.fov_horizontal/2
            xyz = xyz[np.where( np.arctan2(np.abs(xyz[:,1]), np.abs(xyz[:,0])) < max_angle )[0], :]

        elif self.mode =="polar":
            angles = np.arctan2(xyz[:,1],xyz[:,0])
            radius = np.linalg.norm(xyz[:,0:1],axis=1)

            mask = (np.abs(angles) < self.fov_horizontal/2) & (radius < self.r_max) & (radius > self.r_min)
            xyz = xyz[mask,:]

        else:
            raise NotImplementedError(f"Choose as mode 'linear' 'log' or 'polar' not {self.mode}")
            

        voxels = self.voxelizer([[torch.from_numpy(xyz)]])
        

        if self.return_original:
            return dict(voxels=voxels, pointcloud=pointcloud)
        return voxels

    def __len__(self):
        return len(self.dataset)
    

class RangeImage(Dataset):
    def __init__(self, dataset, config, is_train=True, dataset_type="nuscenes"):
        self.dataset = dataset  # e.g. the Nuscenes object
        self.config = config['model']['params']['lossconfig']['params']['dataset_config']
        self.is_train = is_train
        self.dataset_type = dataset_type
        
        self.proj_h = self.config['size'][0]
        self.proj_w = self.config['size'][1]
        self.fov = self.config['fov']
        self.depth_range = self.config['depth_range']
        self.log_scale = self.config['log_scale']
        if self.log_scale:
            self.depth_scale = self.config['depth_scale']
        else:
            self.depth_scale = 2**self.config['depth_scale']

        if dataset_type=="frontcam":
            self.t_ratio = self.config['t_ratio']
            self.b_ratio = self.config['b_ratio']
            self.l_ratio = self.config['l_ratio']
            self.r_ratio = self.config['r_ratio']
        self.sensor_pos = self.config['sensor_pos']

        if self.log_scale:
            self.depth_thresh = (np.log2(1./255. + 1) / self.depth_scale) * 2. - 1 + 1e-6
        else:
            self.depth_thresh = (1./255. / self.depth_scale) * 2. - 1 + 1e-6

        if self.is_train and 'augmentation' in self.config:
            augment_config = self.config['augmentation']
            self.augmentor = get_augmentor(augment_config)
            
        else:
            self.augmentor = None

    def __getitem__(self, index):
        pointcloud = self.dataset.loadDataByIndex(index)

        #if self.is_train:
        #    pointcloud = self.augmentor.doAugmentation(pointcloud)
            
        pointcloud_xyz = pointcloud[:, :3]

        # Move sensor to the real world position on the car.
        pointcloud_xyz -= self.sensor_pos

        if self.dataset_type in ['nuscenes','kitti']:
            proj_range = pcd2range(pointcloud_xyz, [self.proj_h, self.proj_w], self.fov, self.depth_range)
            proj_range, proj_mask = process_scan(proj_range, self.log_scale, self.depth_scale, self.depth_thresh)

        elif self.dataset_type == 'frontcam':
            range_img = pcd2range(pointcloud_xyz, [self.proj_h, self.proj_w], self.fov, self.depth_range)

            t_ratio = int(round(self.t_ratio*self.proj_h))
            b_ratio = int(round(self.b_ratio*self.proj_h))
            l_ratio = int(round(self.l_ratio*self.proj_w))
            r_ratio = int(round(self.r_ratio*self.proj_w))

            range_img = range_img[t_ratio:b_ratio, l_ratio:r_ratio]

            proj_range, proj_mask = process_scan(range_img, self.log_scale, self.depth_scale, self.depth_thresh)

        else:
            raise NotImplementedError(f"dataset_type '{self.dataset_type}' is not a valid option")
        
        

        #proj_range_tensor = torch.from_numpy(proj_range)
        #proj_range_tensor = proj_range_tensor.unsqueeze(0)  # My addition to fix wrong dimensions error in convolution

        output = {
            'input2d': proj_range,
            'mask': proj_mask,
        }

        if self.dataset_type == 'frontcam':
            output['ratios'] = (int(self.proj_h), int(self.proj_w), (int(t_ratio), int(b_ratio), int(l_ratio), int(r_ratio)))

        return output

    def __len__(self):
        return len(self.dataset)
    

def get_augmentor(augment_config):
    augment_params = AugmentParams()
    augment_params.setFlipProb(
        p_flipx=augment_config['p_flipx'], p_flipy=augment_config['p_flipy'])
    augment_params.setTranslationParams(
        p_transx=augment_config['p_transx'], trans_xmin=augment_config[
            'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
        p_transy=augment_config['p_transy'], trans_ymin=augment_config[
            'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
        p_transz=augment_config['p_transz'], trans_zmin=augment_config[
            'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
    augment_params.setRotationParams(
        p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
            'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
        p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
            'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
        p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
            'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
    if 'p_scale' in augment_config:
        augment_params.sefScaleParams(
            p_scale=augment_config['p_scale'],
            scale_min=augment_config['scale_min'],
            scale_max=augment_config['scale_max'])
        print(f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
    return Augmentor(augment_params)