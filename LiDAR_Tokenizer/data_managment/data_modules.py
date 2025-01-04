try:
    from data_managment import datasets, nuscenes, kitti, frontcam
except Exception as e:
    print(e)
    print("\nTRYING WITH RELATIVE IMPORTS, BUT IF THE ISSUE WAS SOMETHING ELSE THEN YOU ARE PROBABLY MISSING A PACKAGE\n")   
    from ..data_managment import datasets, nuscenes, kitti, frontcam

from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
import yaml
import os
import torch

def collate_voxel(batch):
    if isinstance(batch[0], dict):
        batch_out = dict()
        batch_out['voxels'] = torch.stack([batch[i]['voxels'] for i in range(len(batch))])
        batch_out['pointcloud'] = [batch[i]['pointcloud'] for i in range(len(batch))]
        return batch_out
    else:
        return torch.stack(batch)

class LidarTokenizerModule(LightningDataModule):
    def __init__(self,
                 dataroot='/mnt/data/Public_datasets/nuScenes/',
                 batch_size=1,
                 num_workers=4,
                 info_file_path="data_managment/nuscenes_info.json",
                 root_dir="/home/LiDAR-Tokenizer",
                 mode="range_image",
                 dataset="nuscenes",
                 config_path=None,
                 version="v1.0-trainval"):
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.dataset_name = dataset

        if dataset=="nuscenes":
            train_dataset = nuscenes.Nuscenes(self.dataroot, version=version, split='train', info_file_path=info_file_path)
            val_dataset = nuscenes.Nuscenes(self.dataroot, version=version, split='val', info_file_path=info_file_path)
            test_dataset = nuscenes.Nuscenes(self.dataroot, version=version, split='test', info_file_path=info_file_path)
        elif dataset=="semantic_kitti":
            print("Warning: Kitti might require some adjustments as it has not been used in the experiments so it was not updated.")
            train_dataset = kitti.Kitti(split='train', info_file_path=info_file_path)
            val_dataset = kitti.Kitti(split='val', info_file_path=info_file_path)
        elif dataset=="frontcam":
            train_dataset = frontcam.FrontCam(self.dataroot, version=version, split='train', info_file_path=info_file_path, channels="all")
            val_dataset = frontcam.FrontCam(self.dataroot, version=version, split='val', info_file_path=info_file_path, channels="all")
            test_dataset = frontcam.FrontCam(self.dataroot, version=version, split='test', info_file_path=info_file_path, channels="all")
        else:
            raise NotImplementedError("Choose as dataset one of: 'nuscenes', 'semantic_kitti', 'frontcam'.")
        
        with open(os.path.join(root_dir, config_path), 'r') as file:
            self.data_config = yaml.safe_load(file)

        if mode=="range_image":
            self.train_dataset = datasets.RangeImage(train_dataset, config=self.data_config, is_train=True, dataset_type=dataset)
            self.val_dataset = datasets.RangeImage(val_dataset, config=self.data_config, is_train=False, dataset_type=dataset)
            self.test_dataset = datasets.RangeImage(test_dataset, config=self.data_config, is_train=False, dataset_type=dataset)
        elif mode=="voxel":
            self.train_dataset = datasets.Voxels(train_dataset, config=self.data_config, is_train=True)
            self.val_dataset = datasets.Voxels(val_dataset, config=self.data_config, is_train=False)
            self.test_dataset = datasets.Voxels(test_dataset, config=self.data_config, is_train=False, return_original=True)
        else:
            raise NotImplementedError("Choose mode as one of: 'range_image', 'voxel'.")
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.mode=="range_image":
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        elif self.mode=="voxel":
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_voxel)

    def val_dataloader(self):
        if self.mode=="range_image":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        elif self.mode=="voxel":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_voxel)

    def test_dataloader(self):
        if self.mode=="range_image":
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.mode=="voxel":
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_voxel)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage):
        pass

