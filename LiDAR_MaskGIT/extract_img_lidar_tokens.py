from LiDAR_Tokenizer.frontcam_match_timestamps import match_timestamps
from LiDAR_Tokenizer.data_managment.preprocess_frontcam import preprocess_range, preprocess_voxel
from LiDAR_Tokenizer.model_managment.helpers import load_model
import os 
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import json

from Network.Taming.models.vqgan import VQModel
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

def main(args):

    data_folder = args.data_folder

    img_dirs = [
        os.path.join(data_folder, "2024-09-06-07-42-41/camera1"),
        os.path.join(data_folder, "2024-09-06-10-59-33/camera1"),
        os.path.join(data_folder, "2024-09-06-11-19-38/camera1"),
        os.path.join(data_folder, "2024-09-06-11-42-34/camera1"),
        os.path.join(data_folder, "2024-09-09-12-33-35/camera1"),
        os.path.join(data_folder, "2024-09-13-06-43-51/camera1"),
        os.path.join(data_folder, "2024-09-13-07-06-04/camera1"),
        os.path.join(data_folder, "2024-09-13-10-31-13/camera1"),
        os.path.join(data_folder, "2024-09-13-12-19-26/camera1"),
        os.path.join(data_folder, "2024-09-17-06-52-07/camera1")
        ]

    lidar_dirs = [
        os.path.join(data_folder, "npz_2024-09-06-07-42-41/"),
        os.path.join(data_folder, "npz_2024-09-06-10-59-33/"),
        os.path.join(data_folder, "npz_2024-09-06-11-19-38/"),
        os.path.join(data_folder, "npz_2024-09-06-11-42-34/"),
        os.path.join(data_folder, "npz_2024-09-09-12-33-35/"),
        os.path.join(data_folder, "npz_2024-09-13-06-43-51/"),
        os.path.join(data_folder, "npz_2024-09-13-07-06-04/"),
        os.path.join(data_folder, "npz_2024-09-13-10-31-13/"),
        os.path.join(data_folder, "npz_2024-09-13-12-19-26/"),
        os.path.join(data_folder, "npz_2024-09-17-06-52-07/")
        ]
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Load image model.
    vqgan_folder = args.vqgan_folder 
    patch_size = 16

    config = OmegaConf.load(vqgan_folder + "model.yaml")
    img_model = VQModel(**config.model.params)
    checkpoint = torch.load(vqgan_folder + "last.ckpt", map_location=device)["state_dict"]
    img_model.load_state_dict(checkpoint, strict=False)
    img_model = img_model.eval()
    img_model = img_model.to(device)

    # range image lidar tokenizer
    lidar_config_range_image = args.range_image_config 
    lidar_model_range_image = load_model(lidar_config_range_image)
    checkpoint_range_image = torch.load(args.range_image_model, map_location=device)
    lidar_model_range_image.load_state_dict(checkpoint_range_image["state_dict"])
    lidar_model_range_image.eval()
    lidar_model_range_image = lidar_model_range_image.to(device)

    # voxel lidar tokenizer
    lidar_config_voxel = args.voxel_config
    lidar_model_voxel = load_model(lidar_config_voxel)
    checkpoint_voxel = torch.load(args.voxel_model, map_location=device)
    lidar_model_voxel.load_state_dict(checkpoint_voxel["state_dict"])
    lidar_model_voxel.eval()
    lidar_model_voxel = lidar_model_voxel.to(device)

    for img_dir,lidar_dir in tqdm(zip(img_dirs,lidar_dirs)):

        # Load matching images and point clouds.
        pcd_fns = sorted(os.listdir(lidar_dir))
        pcd_ts = []
        for pcd_fn in tqdm(pcd_fns):
            pcd = np.load(os.path.join(lidar_dir, pcd_fn))
            ts = float(pcd['timestamp'])
            pcd_ts.append(ts)

        matching_img_fns, match_time_diff = match_timestamps(pcd_ts, img_dir)
        mask_inds = np.where(match_time_diff < 0.1)

        pcd_fns = [os.path.join(lidar_dir,fn) for fn in pcd_fns]
        pcd_fns = np.array(pcd_fns)[mask_inds]
        pcd_ts = np.array(pcd_ts)[mask_inds]
        matching_img_fns = np.array(matching_img_fns)[mask_inds]

        # split because of OOM issues
        splits = int(len(pcd_fns)/80)
        pcd_fns = np.array_split(pcd_fns, splits)
        pcd_ts = np.array_split(pcd_ts,splits)
        matching_img_fns = np.array_split(matching_img_fns,splits)

        for j in range(len(pcd_fns)):
            pcd_fns_batch = pcd_fns[j]
            pcd_ts_batch = pcd_ts[j]
            matching_img_fns_batch = matching_img_fns[j]

            imgs = [Image.open(image_path).convert('RGB') for image_path in tqdm(matching_img_fns_batch)]
            imgs = [(transform(img) *2 -1).type(torch.float32) for img in imgs]

            # Tokenize images
            image_codes = []

            image_batches = torch.stack(imgs)
            image_batches = torch.split(image_batches,4)
            for image_batch in tqdm(image_batches):
                b,c,h,w = image_batch.shape
                with torch.no_grad():
                    emb, _, [_, _, code] = img_model.encode(image_batch.to(device))
                    code = code.reshape(b, h//patch_size, w//patch_size)
                    image_codes += list(code)

            del imgs, image_batches

            # Tokenize point clouds
            lidar_codes_range_image = []
            lidar_codes_voxel = []

            pcds_preprocessed_range_image = preprocess_range(pcd_fns_batch, lidar_config_range_image)
            pcd_batches_range_image = torch.from_numpy(pcds_preprocessed_range_image)
            pcd_batches_range_image = torch.split(pcd_batches_range_image,4)

            pcds_preprocessed_voxel = preprocess_voxel(pcd_fns_batch, lidar_config_voxel)
            pcd_batches_voxel = torch.split(pcds_preprocessed_voxel,4)

            for pcd_batch in tqdm(pcd_batches_range_image):
                with torch.no_grad():
                    _, _, ind = lidar_model_range_image.encode(pcd_batch.to(device))
                    lidar_codes_range_image += list(ind)

            for pcd_batch in tqdm(pcd_batches_voxel):
                with torch.no_grad():
                    _, _, ind = lidar_model_voxel.encode(pcd_batch.to(device))
                    ind = ind.reshape(pcd_batch.shape[0],-1)
                    ind = ind.to(torch.int)
                    lidar_codes_voxel += list(ind)


            del pcds_preprocessed_range_image, pcds_preprocessed_voxel, pcd_batches_voxel, pcd_batches_range_image
        
            folder_name = img_dir.split('/')[-2]
            
            if not os.path.exists(os.path.join(data_folder, f"img_tokens_{folder_name}")):
                os.makedirs(os.path.join(data_folder, f"img_tokens_{folder_name}"))
            
            if not os.path.exists(os.path.join(data_folder, f"lidar_tokens_{folder_name}")):
                os.makedirs(os.path.join(data_folder, f"lidar_tokens_{folder_name}"))
            
            if not os.path.exists(os.path.join(data_folder, f"lidar_tokens_voxel_{folder_name}")):
                os.makedirs(os.path.join(data_folder, f"lidar_tokens_voxel_{folder_name}"))

            for i,ts in enumerate(pcd_ts_batch):
                torch.save(image_codes[i], os.path.join(data_folder, f"img_tokens_{folder_name}/img_tokens_{ts}.pt"))
                torch.save(lidar_codes_range_image[i], os.path.join(data_folder, f"lidar_tokens_{folder_name}/lidar_tokens_{ts}.pt"))
                torch.save(lidar_codes_voxel[i], os.path.join(data_folder, f"lidar_tokens_voxel_{folder_name}/lidar_tokens_voxel_{ts}.pt"))

            # Image file & tokens match dictionary
            img_paths = {f"img_tokens_{folder_name}/img_tokens_{pcd_ts_batch[k]}.pt": matching_img_fns_batch[k] for k in range(len(matching_img_fns_batch))}

            if os.path.exists(os.path.join(data_folder, f"img_paths_{folder_name}.json")):
                with open(os.path.join(data_folder, f"img_paths_{folder_name}.json"), "r") as json_file:
                    existing_image_paths = json.load(json_file)
                    img_paths.update(existing_image_paths)

            with open(os.path.join(data_folder, f"img_paths_{folder_name}.json"), "w+") as json_file: 
                json.dump(img_paths, json_file)

            # Lidar file & tokens match dictionary
            lidar_paths = {f"lidar_tokens_{folder_name}/lidar_tokens_{pcd_ts_batch[k]}.pt": pcd_fns_batch[k] for k in range(len(pcd_fns_batch))}

            if os.path.exists(os.path.join(data_folder, f"lidar_paths_{folder_name}.json")):
                with open(os.path.join(data_folder, f"lidar_paths_{folder_name}.json"), "r") as json_file:
                    existing_lidar_paths = json.load(json_file)
                    lidar_paths.update(existing_lidar_paths)

            with open(os.path.join(data_folder, f"lidar_paths_{folder_name}.json"), "w+") as json_file: 
                json.dump(lidar_paths, json_file)

            lidar_paths_voxel = {f"lidar_tokens_voxel_{folder_name}/lidar_tokens_voxel_{pcd_ts_batch[k]}.pt": pcd_fns_batch[k] for k in range(len(pcd_fns_batch))}

            if os.path.exists(os.path.join(data_folder, f"lidar_paths_voxel_{folder_name}.json")):
                with open(os.path.join(data_folder, f"lidar_paths_voxel_{folder_name}.json"), "r") as json_file:
                    existing_lidar_paths = json.load(json_file)
                    lidar_paths_voxel.update(existing_lidar_paths)

            with open(os.path.join(data_folder, f"lidar_paths_voxel_{folder_name}.json"), "w+") as json_file: 
                json.dump(lidar_paths_voxel, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder",        type=str, default="/data", help="folder containing the dataset")
    parser.add_argument("--vqgan-folder",       type=str, default="LiDAR_MaskGIT/pretrained_maskgit/VQGAN/", help="folder of the pretrained VQGAN")
    parser.add_argument("--range-image-config", type=str, default="LiDAR_MaskGIT/LiDAR_Tokenizer/model_managment/final_configs/frontcam/range_image/config_FINAL_FRONTCAM_range_fsq.yaml",help="range image lidar tokenizer config path")
    parser.add_argument("--range-image-model",  type=str, default="LiDAR_MaskGIT/pretrained_maskgit/LiDARTokenizer/ld_final/range_model.ckpt",help="range image lidar tokenizer trained model path")
    parser.add_argument("--voxel-config",       type=str, default="LiDAR_MaskGIT/LiDAR_Tokenizer/model_managment/final_configs/frontcam/voxel/config_FINAL_FRONTCAM_voxel_log_final.yaml",help="voxel lidar tokenizer config path")
    parser.add_argument("--voxel-model",        type=str, default="LiDAR_MaskGIT/pretrained_maskgit/LiDARTokenizer/ul_final/voxel_model.ckpt",help="voxel lidar tokenizer trained model path")
    args = parser.parse_args()

    main(args)
