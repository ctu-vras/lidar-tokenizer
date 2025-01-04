import matplotlib.image as mpimg
from PIL import Image

import matplotlib
import numpy as np
#import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib.colors import ListedColormap



def plot_images(img, range_img, fn):
    fig = plt.figure(figsize=(12, 6))

    # Define grid layout
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    # First subplot: image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, aspect='equal')
    ax1.set_title('Image')
    ax1.set_axis_off()

    cmap = plt.cm.coolwarm  # Start with the 'coolwarm' colormap
    new_cmap = cmap(np.linspace(0, 1, cmap.N))  # Get the color range
    new_cmap[0, :] = [0, 0, 0, 1]  # Set the lowest color to black (RGBA format)
    custom_cmap = ListedColormap(new_cmap)

    # Second subplot: range image
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(range_img, aspect='equal', cmap=custom_cmap, vmin=0, vmax=165)
    ax2.set_title('Range image')
    ax2.set_axis_off()

    #plt.subplots_adjust(right=0.85)

    #cbar_ax = fig.add_axes([1, 0.388, 0.015, 0.55])  # [left, bottom, width, height]
    #cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), orientation='vertical')
    cbar.set_label("[m]")

    #plt.tight_layout()
    plt.savefig(fn)
    plt.close()


if __name__ == "__main__":
    img = np.random.rand(410,490,3)
    range_img = np.random.rand(96,192)
    plot_images(img, range_img, "LiDAR_MaskGIT/saved_img/test.png")