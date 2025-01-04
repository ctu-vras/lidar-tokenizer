import torch
import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class PolarVoxelizer(torch.nn.Module):
    def __init__(self,
                 z_min=-2,
                 z_max=18,
                 z_step=0.2,
                 fov=2.268,  #130 degs in rad
                 num_angle_bins=192,
                 r_min = 2.7,
                 r_max = 165,
                 num_r_bins = 320,
                 force_num_r_bins = True,
                 r_axis_spacing = 'log',    # or linear
                 ):
        
        super().__init__()

        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step

        self.z_depth = round((self.z_max - self.z_min) / self.z_step)

        self.fov = fov

        self.num_angle_bins = num_angle_bins
        self.r_min = r_min
        self.r_max = r_max
        self.num_r_bins = num_r_bins

        self.angle_bins = torch.linspace(-self.fov/2, self.fov/2, self.num_angle_bins)

        angle_delta = self.fov/self.num_angle_bins

        def get_kth_bin(r_min, delta, k):
            return r_min*((1+delta)**k)
        
        def get_kmax(r_min, r_max, delta):
            return math.ceil(math.log(r_max, 1+delta) - math.log(r_min, 1+delta)) +1
        
        if r_axis_spacing == "linear":
            self.r_bins = torch.linspace(self.r_min, self.r_max, self.num_r_bins)

        elif r_axis_spacing == "log":
            #self.r_bins = torch.logspace(math.log(self.r_min), math.log(self.r_max), self.num_r_bins, math.e)
            if force_num_r_bins:
                kmax = self.num_r_bins - 1
                angle_delta = ((self.r_max+1e-4)/self.r_min)**(1/kmax)-1
                self.r_bins = torch.Tensor([get_kth_bin(r_min, angle_delta, k) for k in range(kmax+1)])
                self.num_r_bins = len(self.r_bins)
            else:
                self.r_bins = torch.Tensor([get_kth_bin(r_min, angle_delta, k) for k in range(get_kmax(r_min, r_max, angle_delta))])
                self.num_r_bins = len(self.r_bins)
                print(f"PolarVoxelizer: Selected log spacing, desired num_r_bins={num_r_bins} will be ignored. Instead using num_r_bins={self.num_r_bins}.")

        else:
            raise NotImplementedError(f"for r_axis_spacing choose 'linear' or 'log', not {r_axis_spacing}")
        
    def voxels2points(self, voxels):
        voxels = voxels.cpu()
        non_zero_indices = torch.nonzero(voxels)
        x_grid_coords = non_zero_indices[:,2]
        y_grid_coords = non_zero_indices[:,1]
        z_grid_coords = non_zero_indices[:,0]

        zero = torch.zeros(1)
        angles = (self.angle_bins[y_grid_coords] + self.angle_bins[torch.maximum(zero, y_grid_coords-1).to(dtype=int)])/2
        radius = (self.r_bins[x_grid_coords] + self.r_bins[torch.maximum(zero, x_grid_coords-1).to(dtype=int)])/2

        x = radius*torch.cos(angles)
        y = radius*torch.sin(angles)

        z = ((z_grid_coords+0.5) * self.z_step) + self.z_min
        xyz = torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)], dim=1)
        return xyz

    def voxelize_single(self, pointcloud, bev):

        angles = torch.atan2(pointcloud[:,1],pointcloud[:,0])
        radius = torch.sqrt(torch.sum(pointcloud[:,0:2]**2, dim=1))

        mask = (torch.abs(angles) < self.fov/2) & (radius < self.r_max) & (radius > self.r_min)

        angles = angles[mask]
        radius = radius[mask]

        x_grid_coords = torch.bucketize(radius.contiguous(), self.r_bins)
        y_grid_coords = torch.bucketize(angles.contiguous(), self.angle_bins)
        z_grid_coords = torch.floor((pointcloud[mask, 2] - self.z_min) / self.z_step).long()

        bev[z_grid_coords, y_grid_coords, x_grid_coords] = 1.0

    def forward(self, lidars):
        batch_size = len(lidars)
        assert batch_size > 0 and len(lidars[0]) > 0
        num_sweep = len(lidars[0])

        bev = torch.zeros(
            (batch_size, num_sweep, self.z_depth, self.num_angle_bins, self.num_r_bins),
            dtype=torch.float,
            device=lidars[0][0][0].device,
        )

        for b in range(batch_size):
            assert len(lidars[b]) == num_sweep
            for i in range(num_sweep):
                self.voxelize_single(lidars[b][i], bev[b][i])
        return bev.view(batch_size, num_sweep * self.z_depth, self.num_angle_bins, self.num_r_bins)[0]
    
class LogVoxelizer(torch.nn.Module):
    def __init__(self,
                 x_min=2.7,
                 x_max=165,
                 num_x_bins=320,
                 num_angle_bins=192,
                 z_min=-2,
                 z_max=18,
                 z_step=0.2,
                 fov=2.268  #130 degs in rad
                 ):
        
        super().__init__()

        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.fov = fov
        self.num_x_bins = num_x_bins
        self.num_angle_bins = num_angle_bins

        self.z_depth = round((self.z_max - self.z_min) / self.z_step)

        self.angle = math.pi/2-self.fov/2      # fov in rad

        #self.x_bins = torch.linspace(self.x_min, self.x_max, self.xy_size)
        self.x_bins = torch.logspace(math.log(self.x_min), math.log(x_max), self.num_x_bins, math.e)
        self.edges = self.x_bins/math.tan(self.angle) #[x_bin/np.tan(self.angle) for x_bin in self.x_bins]
        #self.y_bins = [np.linspace(-edge,edge,self.grid_size) for edge in self.edges]
    
    def voxels2points(self, voxels):
        voxels = voxels.cpu()
        non_zero_indices = torch.nonzero(voxels)
        x_grid_coords = non_zero_indices[:,2]
        y_grid_coords = non_zero_indices[:,1]
        z_grid_coords = non_zero_indices[:,0]

        x = (self.x_bins[x_grid_coords] + self.x_bins[torch.maximum(torch.zeros(1),x_grid_coords-1).to(int)])/2

        edges = self.edges[x_grid_coords]
        y = (y_grid_coords+0.5)*2*edges/self.num_angle_bins - edges

        z = ((z_grid_coords+0.5) * self.z_step) + self.z_min
        xyz = torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)], dim=1)
        return xyz

    def voxelize_single(self, pointcloud, bev):

        #x_log = torch.log(pointcloud[:,0] - self.x_min)

        # x
        x_grid_coords = torch.bucketize(pointcloud[:,0].contiguous(), self.x_bins)
        # y
        edges = self.edges[x_grid_coords]
        y_grid_coords = torch.floor((pointcloud[:,1] + edges)*(self.num_angle_bins/(2*edges))).long()
        # z
        z_grid_coords = torch.floor((pointcloud[:, 2] - self.z_min) / self.z_step).long()

        bev[z_grid_coords, y_grid_coords, x_grid_coords] = 1.0

    def forward(self, lidars):
        batch_size = len(lidars)
        assert batch_size > 0 and len(lidars[0]) > 0
        num_sweep = len(lidars[0])

        bev = torch.zeros(
            (batch_size, num_sweep, self.z_depth, self.num_angle_bins, self.num_x_bins),
            dtype=torch.float,
            device=lidars[0][0][0].device,
        )

        for b in range(batch_size):
            assert len(lidars[b]) == num_sweep
            for i in range(num_sweep):
                self.voxelize_single(lidars[b][i], bev[b][i])
        return bev.view(batch_size, num_sweep * self.z_depth, self.num_angle_bins, self.num_x_bins)[0]


class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, x_min, x_max, y_min, y_max, step, z_min, z_max, z_step, fov=None):
        super().__init__()

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step

        self.width = round((self.x_max - self.x_min) / self.step)
        self.height = round((self.y_max - self.y_min) / self.step)
        self.z_depth = round((self.z_max - self.z_min) / self.z_step)
        self.depth = self.z_depth

    def voxels2points(self, voxels):
        non_zero_indices = torch.nonzero(voxels)
        x_grid_coords = non_zero_indices[:,2]
        y_grid_coords = non_zero_indices[:,1]
        z_grid_coords = non_zero_indices[:,0]

        x = (x_grid_coords+0.5) * self.step + self.x_min
        y = (y_grid_coords+0.5) * self.step + self.y_min
        z = (z_grid_coords+0.5) * self.z_step + self.z_min
        xyz = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)

        return xyz

    def voxelize_single(self, lidar, bev):
        """Voxelize a single lidar sweep into image frame
        Image frame:
        1. Increasing depth indices corresponds to increasing real world z
            values.
        2. Increasing height indices corresponds to decreasing real world y
            values.
        3. Increasing width indices corresponds to increasing real world x
            values.
        Args:
            lidar (torch.Tensor N x 4 or N x 5) x, y, z, intensity, height_to_ground (optional)
            bev (torch.Tensor D x H x W) D = depth, the bird's eye view
                raster to populate
        """
        # assert len(lidar.shape) == 2 and (lidar.shape[1] == 4 or lidar.shape[1] == 5) and lidar.shape[0] > 0
        # indices_h = torch.floor((self.y_max - lidar[:, 1]) / self.step).long()
        indices_h = torch.floor((lidar[:, 1] - self.y_min) / self.step).long()
        indices_w = torch.floor((lidar[:, 0] - self.x_min) / self.step).long()
        indices_d = torch.floor((lidar[:, 2] - self.z_min) / self.z_step).long()

        valid_mask = ~torch.any(
            torch.stack(
                [
                    indices_h < 0,
                    indices_h >= self.height,
                    indices_w < 0,
                    indices_w >= self.width,
                    indices_d < 0,
                    indices_d >= self.z_depth,
                ]
            ),
            dim=0,
        )
        indices_h = indices_h[valid_mask]
        indices_w = indices_w[valid_mask]
        indices_d = indices_d[valid_mask]
        # 4. Assign indices to 1
        bev[indices_d, indices_h, indices_w] = 1.0

    def forward(self, lidars):
        """Voxelize multiple sweeps in the current vehicle frame into voxels
            in image frame
        Args:
            list(list(tensor)): B * T * tensor[N x 4],
                where B = batch_size, T = 5, N is variable,
                4 = [x, y, z, intensity]
        Returns:
            tensor: [B x D x H x W], B = batch_size, D = T * depth, H = height,
                W = width
        """
        batch_size = len(lidars)
        assert batch_size > 0 and len(lidars[0]) > 0
        num_sweep = len(lidars[0])

        bev = torch.zeros(
            (batch_size, num_sweep, self.depth, self.height, self.width),
            dtype=torch.float,
            device=lidars[0][0][0].device,
        )

        for b in range(batch_size):
            assert len(lidars[b]) == num_sweep
            for i in range(num_sweep):
                self.voxelize_single(lidars[b][i], bev[b][i])
        return bev.view(batch_size, num_sweep * self.depth, self.height, self.width)[0]

if __name__ == "__main__":
    vox = Voxelizer(-50., 50., -50., 50., 0.15625, -5., 3., 0.15)

    lidar_path = "/mnt/data/Public_datasets/nuScenes/samples/LIDAR_TOP/n008-2018-08-30-10-33-52-0400__LIDAR_TOP__1535639756450885.pcd.bin"
    raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
    pointcloud = raw_data[:, :4]
    pointcloud = torch.from_numpy(pointcloud)

    xyz = pointcloud[:, :3]
    intensity = pointcloud[:, 3:4]
    xyz = torch.cat((torch.clamp(xyz[:,:2], -50, 50), torch.clamp(xyz[:,2:3], -5., 3.), intensity), dim=1)
    #xyz = torch.unsqueeze(xyz, 0)
    xyz_vox = vox([[xyz]])

    voxel_grid = xyz_vox[0].numpy()

    # Get the coordinates of non-zero voxels (value = 1)
    voxels = np.argwhere(voxel_grid == 1)

    # Separate x, y, z coordinates for plotting
    z, y, x = voxels.T



    import plotly.graph_objects as go
    import plotly.io as pio

    # Set default renderer
    pio.renderers.default = 'browser'

    # Determine min and max values for the axes to set symmetrical scaling
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Find the range that spans the largest extent to ensure symmetrical scaling
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Calculate axis limits to center the data around the same origin
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    # Set limits based on the largest range
    axis_limits = [x_mid - max_range / 2, x_mid + max_range / 2]

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=2, color='blue')
    )])

    # Update layout to enforce symmetric axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=axis_limits, title="X Axis"),
            yaxis=dict(nticks=10, range=axis_limits, title="Y Axis"),
            zaxis=dict(nticks=10, range=axis_limits, title="Z Axis"),
            aspectmode='cube'  # Ensures all axes are scaled equally
        )
    )

    # Show plot
    fig.show()
