# %%
import importlib
#importlib.reload(my_module)

from data_managment.data_modules import LidarTokenizerModule
from model_managment.helpers import load_model

from PIL import Image
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import yaml
import os
from matplotlib.colors import ListedColormap

cmap = plt.cm.coolwarm  # Start with the 'coolwarm' colormap
new_cmap = cmap(np.linspace(0, 1, cmap.N))  # Get the color range
new_cmap[0, :] = [0, 0, 0, 1]  # Set the lowest color to black (RGBA format)
custom_cmap = ListedColormap(new_cmap)

plt.rcParams.update({'font.size': 18})


root_dir = os.path.dirname(__file__)
config_path = os.path.join(root_dir, "model_managment/config_FINAL_FRONTCAM_ld_fsq_sensor_pos.yaml")

model = load_model(config_path)
checkpoint = torch.load(os.path.join(root_dir, "slurm_logs/ld_FINAL_frontcam_fsq_sensor_pos_241125_1646/models/lidar-diffusion-epoch=39-step=79999.ckpt"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

data = LidarTokenizerModule(dataroot = "/data",
                      batch_size = 1,
                      num_workers = 2,
                      info_file_path = os.path.join(root_dir, "data_managment/frontcam_single.json"),    # specifies which files are train/val/test
                      mode="range_image",
                      dataset="frontcam",
                      config_path=config_path,
                      version="v1.0-trainval"
                      ) 

trainer = pl.Trainer(gpus=0)
predictions = trainer.predict(model, datamodule=data)

# %%
print(f" Range img MSE: {round(float(predictions[0]['test_img_mse'][0]),3)}")
print(f" Range img MAE: {round(float(predictions[0]['test_img_mae'][0]),3)}")
print(f"Pointcloud MSE: {round(float(predictions[0]['test_points_mse'][0]),3)}")
print(f"Pointcloud MAE: {round(float(predictions[0]['test_points_mae'][0]),3)}")
#print(f"            Pointcloud (clamped) MSE: {round(float(predictions[0]['test_points_clamped_mse']),3)}")
#print(f"            Pointcloud (clamped) MAE: {round(float(predictions[0]['test_points_clamped_mae']),3)}")

# %%
""" imgs = [
        "/mnt/data/Public_datasets/nuScenes/samples/CAM_FRONT_LEFT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639756404799.jpg",
 "/mnt/data/Public_datasets/nuScenes/samples/CAM_FRONT/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639756412404.jpg",
 "/mnt/data/Public_datasets/nuScenes/samples/CAM_FRONT_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_RIGHT__1535639756420482.jpg",
 "/mnt/data/Public_datasets/nuScenes/samples/CAM_BACK_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_BACK_RIGHT__1535639756428113.jpg",
 "/mnt/data/Public_datasets/nuScenes/samples/CAM_BACK/n008-2018-08-30-10-33-52-0400__CAM_BACK__1535639756437558.jpg",
 "/mnt/data/Public_datasets/nuScenes/samples/CAM_BACK_LEFT/n008-2018-08-30-10-33-52-0400__CAM_BACK_LEFT__1535639756447405.jpg",
 ] """

camera_img = "/data/2024-09-06-11-42-34/camera1/camera1_9054_1725623479706292917.jpg.jpg"
camera_img = mpimg.imread(camera_img)

i = 0

data_config = data.data_config['model']['params']['lossconfig']['params']['dataset_config']
t,b,l,r = predictions[i]['ratios']

x = predictions[i]['x'][0]
x = x*0.5 + 0.5
x = x*data_config['depth_scale']
if data_config['log_scale']:
    x = np.exp2(x) - 1

x_rec = predictions[i]['x_rec'][0] 
x_rec = x_rec*0.5 + 0.5
x_rec = x_rec*data_config['depth_scale']
if data_config['log_scale']:
    x_rec = np.exp2(x_rec) - 1

m = predictions[i]['m'][0]
m_rec = predictions[i]['m_rec'][0]
m_rec[np.argwhere(m_rec<=0)[0], np.argwhere(m_rec<=0)[1]] = -1
m_rec[np.argwhere(m_rec>0)[0], np.argwhere(m_rec>0)[1]] = 1

x_masked = x.copy()
x_masked[np.argwhere(m<0)[0], np.argwhere(m<0)[1]] = -1

x_rec_masked = x_rec.copy()
x_rec_masked[np.argwhere(m_rec<0)[0], np.argwhere(m_rec<0)[1]] = -1

m_rec_err_abs = np.abs(m-m_rec)
x_rec_err_abs = np.abs(x-x_rec)
x_masked_rec_err_abs = np.abs(x_masked-x_rec_masked)

#m_rec_err_rel = np.abs(m-m_rec)/np.abs(m)
#x_rec_err_rel = np.abs(x-x_rec)/np.abs(x)
#x_masked_rec_err_rel = np.abs(x_masked-x_rec_masked)/np.abs(x_masked)

x_rec_err_abs[np.argwhere(m<0)[0], np.argwhere(m<0)[1]] = 0

x = x[t:b,l:r]
m = m[t:b,l:r]

x_rec = x_rec[t:b,l:r]
m_rec = m_rec[t:b,l:r]

x_masked = x_masked[t:b,l:r]

x_rec_masked = x_rec_masked[t:b,l:r]
x_masked_rec_err_abs = x_masked_rec_err_abs[t:b,l:r]

x_rec_err_abs = x_rec_err_abs[t:b,l:r]
m_rec_err_abs = m_rec_err_abs[t:b,l:r]

vmin = -1#min(x.min(), x_rec.min(), x_masked.min(), x_rec_masked.min())
vmax = 1#max(x.max(), x_rec.max(), x_masked.max(), x_rec_masked.max())
#vmax_err = max(x_rec_err_abs.max(), x_masked_rec_err_abs.max())

#fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1, 1, 1])

# Create the subplots for the images
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])
ax5 = plt.subplot(gs[2, 0])
ax6 = plt.subplot(gs[2, 1])
ax7 = plt.subplot(gs[3, 0])
ax8 = plt.subplot(gs[3, 1])

im = ax1.imshow(m, aspect='equal', cmap='Reds', vmin=vmin, vmax=vmax)
ax1.set_title('Original mask')

im2 = ax2.imshow(x, aspect='equal', cmap=custom_cmap, vmin=0, vmax=120)
ax2.set_title('Original image')

#cbar_ax1 = plt.subplot(gs[:, 2])  # The third column for the colorbar
#cbar1 = fig.colorbar(im2, cax=cbar_ax1)
#cbar1.set_label('(m)')

#cbar_ax = fig.add_axes([1, 0.388, 0.015, 0.55])  # [left, bottom, width, height]
#cbar = fig.colorbar(im, cax=cbar_ax)
#cbar.set_label("[m]")

ax3.imshow(m_rec, aspect='equal', cmap='Reds', vmin=vmin, vmax=vmax)
ax3.set_title('Reconstructed mask')

ax4.imshow(x_rec, aspect='equal', cmap=custom_cmap, vmin=0, vmax=120)
ax4.set_title('Reconstructed image')

ax5.imshow(m_rec_err_abs, aspect='equal', cmap='Reds', vmin=0, vmax=1)
ax5.set_title('Mask error')

im6 = ax6.imshow(x_rec_err_abs, aspect='equal', cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=10))
ax6.set_title('Image abs. error')

ax7.imshow(camera_img, aspect='equal')
ax7.set_title('Camera image')
ax7.axis('off')

im8 = ax8.imshow(x_rec_masked, aspect='equal', cmap=custom_cmap, vmin=0, vmax=120)
ax8.set_title('Reconstructed masked image')

#fig.subplots_adjust(hspace=0.3)  # Adjust the vertical space between rows (optional)
#axes[2, 0].plot([0, 1], [0.5, 0.5], color='black', lw=2, transform=fig.transFigure, figure=fig)

#fig.subplots_adjust(hspace=0.3)  # Adjust the vertical space between rows
#axes[2, 0].axhline(y=1, color='black', linewidth=2, transform=fig.transFigure, figure=fig)


cbar_ax1 = plt.subplot(gs[0:2, 2])  # The third column for the colorbar
cbar1 = fig.colorbar(im8, cax=cbar_ax1)
cbar1.set_label('(m)')

cbar_ax2 = plt.subplot(gs[2:3, 2])  # The third column for the colorbar
cbar2 = fig.colorbar(im6, cax=cbar_ax2)
cbar2.set_label('(m)')

#plt.subplots_adjust(right=0.85)

#cbar_ax = fig.add_axes([1, 0.388, 0.015, 0.55])  # [left, bottom, width, height]
#cbar = fig.colorbar(im5, cax=cbar_ax)
#cbar = fig.colorbar(im, ax=axes[0,0], orientation='vertical')
#cbar.set_label("[m]")

#cbar_ax_2 = fig.add_axes([1, 0.06, 0.015, 0.23])  # [left, bottom, width, height]
#cbar_2 = fig.colorbar(im8, cax=cbar_ax_2)
#cbar_2.set_label("[m]") """


plt.tight_layout()
plt.show()
plt.savefig("LiDAR_Tokenizer/plotting_functions/range_frontcam.png")



# 3D plot

i = 0

fig = go.Figure(data=[go.Scatter3d(
    x=predictions[i]['pcd_rec'][0][:,0],
    y=predictions[i]['pcd_rec'][0][:,1],
    z=predictions[i]['pcd_rec'][0][:,2],
    mode='markers',
    marker=dict(size=2, color='blue')
)])

fig.add_trace(go.Scatter3d(
    x=predictions[i]['pcd_rec_masked'][0][:,0],
    y=predictions[i]['pcd_rec_masked'][0][:,1],
    z=predictions[i]['pcd_rec_masked'][0][:,2],
    mode='markers',
    marker=dict(size=2, color='green')  # Set color for the second point cloud
))

fig.add_trace(go.Scatter3d(
    x=predictions[i]['pcd'][0][:,0],
    y=predictions[i]['pcd'][0][:,1],
    z=predictions[i]['pcd'][0][:,2],
    mode='markers',
    marker=dict(size=2, color='red')  # Set color for the second point cloud
))

# ground truth pointcloud
ground_truth_pcd = "/data/npz_2024-09-06-11-42-34/time_1725623479.6972.npz"

raw_data = np.load(ground_truth_pcd)['point_cloud']

all_points = []
for _,channel in enumerate([2,3,4,5,6,7]):
    data_channel = raw_data[np.where(raw_data[:,4] == channel)]

    pcd = data_channel[:,:3]
    all_points.append(pcd)

pointcloud = np.vstack(all_points)
depth = np.linalg.norm(pointcloud,axis=1)
pitch = np.arcsin(pointcloud[:,2] / depth)

mask = np.logical_or(pitch < -5*(np.pi/180), np.logical_and(depth<3.7, np.abs(pointcloud[:,1])<0.9))

ground_truth_pcd = pointcloud[~mask,:]

fig.add_trace(go.Scatter3d(
    x=ground_truth_pcd[:,0],
    y=ground_truth_pcd[:,1],
    z=ground_truth_pcd[:,2],
    mode='markers',
    marker=dict(size=2, color='blue')  # Set color for the second point cloud
))

errors = np.linalg.norm(predictions[i]['pcd_rec_ground_truth_masked'][0] - predictions[i]['pcd'][0],axis=1)
log_errors = np.log10(errors)

cmin = -2.5
cmax = max(log_errors)

tickvals = np.logspace(cmin, cmax, 5)
ticktext = [f"{val:.4f}" for val in tickvals]
actual_tickvals = np.linspace(cmin, cmax,5)

fig.add_trace(go.Scatter3d(
    x=predictions[i]['pcd_rec_ground_truth_masked'][0][:,0],
    y=predictions[i]['pcd_rec_ground_truth_masked'][0][:,1],
    z=predictions[i]['pcd_rec_ground_truth_masked'][0][:,2],
    mode='markers',
    marker=dict(size=3, color=log_errors,  # Use log-transformed errors for coloring
        colorscale='balance',  # Colormap
        colorbar=dict(
            title='Error (log scale, meters)',
            tickvals=actual_tickvals,  # Log scale tick marks
            ticktext=ticktext,
            len=0.7
        ),
        cmin=cmin, 
        cmax=cmax, 
    )
))


all_points = np.concatenate([predictions[i]['pcd_rec_masked'][0], predictions[i]['pcd'][0]], axis=0)

x_min, y_min, z_min = all_points.min(axis=0)
x_max, y_max, z_max = all_points.max(axis=0)

range_min = min(x_min.item(), y_min.item(), z_min.item())
range_max = max(x_max.item(), y_max.item(), z_max.item())

fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=[range_min, range_max]),
        yaxis=dict(nticks=10, range=[range_min, range_max]),
        zaxis=dict(nticks=10, range=[range_min, range_max]),
        aspectmode='cube'
    )
)

#fig.show()
fig.write_html("LiDAR_Tokenizer/plotting_functions/range_frontcam.html")