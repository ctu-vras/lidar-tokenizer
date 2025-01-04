import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from Network.Taming.models.vqgan import VQModel
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def plot_npz(path_to_npz_file):
    ground_truth = np.load(path_to_npz_file)
    ground_truth = ground_truth['arr_0']

    fig = go.Figure(data=[go.Scatter3d(
        x=ground_truth[:,0],
        y=ground_truth[:,1],
        z=ground_truth[:,2],
        mode='markers',
        marker=dict(size=2, color='blue')
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-100, 300]),
            yaxis=dict(nticks=10, range=[-200, 200]),
            zaxis=dict(nticks=10, range=[-100, 300]),
            aspectmode='cube'
        )
    )
    fig.show()

def plot_images(folder_with_images, path_out):
    folder_path = folder_with_images

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    selected_images = random.sample(image_files, min(25, len(image_files)))

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            img_path = os.path.join(folder_path, selected_images[i])
            img = Image.open(img_path)
            
            ax.imshow(img)
            ax.axis('off') 
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(path_out)


def image_tokenizer_check(image_paths):
    """ This file can be used to check how well the ImageNet pretrained VQGAN works on images from other datasets such as FrontCam."""
    imgs = [Image.open(image_path).convert('RGB') for image_path in image_paths]

    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),         
    ])

    imgs = [transform(img).type(torch.float32) * 2 - 1 for img in imgs]

    vqgan_folder = "pretrained_maskgit/VQGAN/"
    device = "cpu"

    config = OmegaConf.load(vqgan_folder + "model.yaml")
    model = VQModel(**config.model.params)
    checkpoint = torch.load(vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
    # Load network
    model.load_state_dict(checkpoint, strict=False)
    model = model.eval()
    model = model.to(device)

    print(f"Size of image tokenizer model: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

    patch_size = 16
    image_batch = torch.stack(imgs)
    b,c,h,w = image_batch.shape
    with torch.no_grad():
        emb, _, [_, _, code] = model.encode(image_batch)

    code = code.reshape(b, h//patch_size, w//patch_size)

    rec_imgs = model.decode_code(torch.clamp(code, 0, 1023))
    rec_imgs = torch.clamp(rec_imgs,-1,1)

    for i in range(b):
        to_pil = transforms.ToPILImage()

        img = to_pil((imgs[i]+1)/2)
        rec_img = to_pil((rec_imgs[i]+1)/2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img)
        axes[0].axis('off')

        axes[1].imshow(rec_img)
        axes[1].axis('off')

        # Display the figure
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_npz("LiDAR_MaskGIT/saved_img/pcd_3.npz")