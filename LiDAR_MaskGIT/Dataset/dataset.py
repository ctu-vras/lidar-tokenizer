from torch.utils.data import Dataset
import json
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

class FrontCamTokens(Dataset):
    def __init__(self, dataroot, version='v1.0-trainval', split='train', info_file_path="frontcam_placeholder_tokens.json", lidar_tokens_offset=1024, device="cpu"):
        assert split in ['train', 'val', 'test']
        self.version = version
        self.split = split
        self.dataroot = dataroot
        self.lidar_tokens_offset = lidar_tokens_offset
        self.device = device

        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        self.filenames = info_data[version][split]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),         
        ])
    
    def __getitem__(self, index):
        img_tokens_path = os.path.join(self.dataroot, self.filenames[index][0])
        lidar_tokens_path = os.path.join(self.dataroot, self.filenames[index][1])
        img_path = os.path.join(self.dataroot, self.filenames[index][2])
        lidar_path = os.path.join(self.dataroot, self.filenames[index][3])

        img_tokens = torch.load(img_tokens_path, map_location="cpu")
        #img_tokens = img_tokens.to(self.device)
        lidar_tokens = torch.load(lidar_tokens_path, map_location="cpu")
        #lidar_tokens = lidar_tokens.to(self.device)

        #lidar_tokens += self.lidar_tokens_offset # NOT USED FOR SEPARATE EMBEDDINGS

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).type(torch.float32)

        return torch.cat((img_tokens.reshape(-1), lidar_tokens)), img, lidar_path

    def __len__(self):
        return len(self.filenames)