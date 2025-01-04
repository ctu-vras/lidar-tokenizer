import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import numpy as np

""" From https://github.com/hancyran/LiDAR-Diffusion """

class GeoConverter(nn.Module):
    def __init__(self, curve_length=4, bev_only=False, dataset_config=dict()):
        super().__init__()
        self.curve_length = curve_length
        self.coord_dim = 3 if not bev_only else 2
        self.convert_fn = self.batch_range2bev if bev_only else self.batch_range2xyz

        fov = dataset_config.fov
        self.fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)  # get field of view total in rad
        
        self.depth_min, self.depth_max = dataset_config.depth_range
        self.log_scale = dataset_config.log_scale

        if self.log_scale:
            self.depth_scale = dataset_config.depth_scale
        else:
            self.depth_scale = 2**dataset_config.depth_scale

        self.size = dataset_config['size']
        #self.t_ratio = int(round(dataset_config.t_ratio * self.size[0]))
        #self.b_ratio = int(round(dataset_config.b_ratio * self.size[0]))
        #self.l_ratio = int(round(dataset_config.l_ratio * self.size[1]))
        #self.r_ratio = int(round(dataset_config.r_ratio * self.size[1]))
        self.register_conversion()

    def register_conversion(self):
        scan_x, scan_y = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
        scan_x = scan_x.astype(np.float64) / self.size[1]
        scan_y = scan_y.astype(np.float64) / self.size[0]

        yaw = (np.pi * (scan_x * 2 - 1))#[self.t_ratio:self.b_ratio, self.l_ratio:self.r_ratio]
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down))#[self.t_ratio:self.b_ratio, self.l_ratio:self.r_ratio]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('cos_yaw', torch.cos(to_torch(yaw)))
        self.register_buffer('sin_yaw', torch.sin(to_torch(yaw)))
        self.register_buffer('cos_pitch', torch.cos(to_torch(pitch)))
        self.register_buffer('sin_pitch', torch.sin(to_torch(pitch)))

    def batch_range2xyz(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_z = self.sin_pitch * batch_depth
        batch_xyz = torch.cat([batch_x, batch_y, batch_z], dim=1)

        return batch_xyz

    def batch_range2bev(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_bev = torch.cat([batch_x, batch_y], dim=1)

        return batch_bev

    def curve_compress(self, batch_coord):
        compressed_batch_coord = F.avg_pool2d(batch_coord, (1, self.curve_length))

        return compressed_batch_coord

    def forward(self, input):
        #input = input / 2. + .5  # [-1, 1] -> [0, 1]    #BUG BUG BUG

        input_coord = self.convert_fn(input)
        if self.curve_length > 1:
            input_coord = self.curve_compress(input_coord)

        return input_coord
    
def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


def square_dist_loss(x, y):
    return torch.sum((x - y) ** 2, dim=1, keepdim=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)