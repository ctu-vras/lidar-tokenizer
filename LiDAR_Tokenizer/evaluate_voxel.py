from data_managment.data_modules import LidarTokenizerModule
from model_managment.helpers import load_model
from data_managment.nuscenes import Nuscenes
from data_managment.datasets import Raw
from data_managment.frontcam import FrontCam

import numpy as np
import torch
import pytorch_lightning as pl

import os
import sys
import argparse

from ignite.metrics import SSIM, PSNR
from ignite.engine.engine import Engine


def main(args):

    root_dir = os.path.dirname(__file__)

    # Save absolute ground truth point clouds
    if args.dataset_name == "nuscenes":
        dataset = Nuscenes(args.dataroot,
                            version='v1.0-trainval',
                            split='test',
                            info_file_path=args.info_file_path)
        
    elif args.dataset_name == "frontcam":
        dataset = FrontCam(args.dataroot,
                        version='v1.0-trainval',
                        split='test',
                        info_file_path=args.info_file_path)

    else:
        raise NotImplementedError(f"invalid dataset_name {args.dataset_name}. choose one of 'nuscenes' or 'frontcam'")
    
    raw_dataset = Raw(dataset)

    ground_truth = []
    for d in raw_dataset:
        ground_truth.append(d)
    np.savez(os.path.join(args.save_npz_dir,'ground_truth.npz'), *ground_truth)
    print(f"Ground truth saved to {os.path.join(args.save_npz_dir,'ground_truth.npz')}")
    del ground_truth

    models = args.models
    names = args.names
    configs = args.configs
    config_paths = [os.path.join(root_dir,config) for config in configs]
    
    assert len(configs) == len(models)
    assert len(configs) == len(names)
    
    voxels_truth_saved = False

    for k in range(len(configs)):
        name = names[k]
        config_path = config_paths[k]
        ckpt = models[k]

        model = load_model(config_path)
        checkpoint = torch.load(os.path.join(root_dir, ckpt))#, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        data = LidarTokenizerModule(dataroot = args.dataroot,
                      batch_size = args.batch_size,
                      num_workers = args.num_workers,
                      info_file_path = args.info_file_path,    # specifies which files are train/val/test
                      root_dir = args.root_dir,
                      mode = "voxel",
                      dataset = args.dataset_name,
                      config_path = config_path
                      ) 

        model.sigma = args.sigma
        trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=0)
        predictions = trainer.predict(model, datamodule=data)

        num_points_in = 0
        num_points_out = 0

        all_iou = torch.Tensor()
        all_codes = torch.Tensor()
        all_codes = all_codes.int()

        for i in range(len(predictions)):

            for b in range(len(predictions[i][3])):
                num_points_in += predictions[i][3][b].shape[0]
                num_points_out += predictions[i][2][b].shape[0]

            codes = predictions[i][4].cpu().reshape(-1)
            iou = predictions[i][0]['test/lidar_rec_iou'].cpu().reshape(-1,1)
            
            all_iou = torch.cat((all_iou, iou))
            all_codes = torch.cat((all_codes, codes))

        print(f"num points in mean: {num_points_in/(len(predictions)*args.batch_size)}")
        print(f"num points out mean: {num_points_out/(len(predictions)*args.batch_size)}")
        top_k = 10
        print(f"sigma {model.sigma}")
        print(f"model: {ckpt}")
        print(f"config: {config_path}")
        print(f"IOU on test dataset of {len(all_iou)} samples: {all_iou.mean()}")
        print(f"Codebook utilization on test dataset of {len(all_iou)*args.batch_size} samples and {model.n_embed} codes: {len(all_codes.unique())/model.n_embed}")
        print(f"Codebook uniformity (top {top_k}) on test dataset of {len(all_iou)*args.batch_size} samples: {all_codes.bincount().sort().values[-top_k:].sum()/len(all_codes)}")
        print(f"Codebook uniformity (top 1) on test dataset of {len(all_iou) * args.batch_size} samples: {all_codes.bincount().sort().values[-1:].sum()/len(all_codes)}")
        print(f"Top 10 codes' counts on test dataset of {len(all_iou) * args.batch_size} samples: {all_codes.bincount().sort().values[-10:]}")


        reconstructions = []

        for i in range(len(predictions)):
            for b in range(len(predictions[i][2])):    # batch
                reconstructions.append(predictions[i][2][b].cpu())

        np.savez(os.path.join(args.save_npz_dir,f'reconstructions_voxel_{name}.npz'), *reconstructions)
        print(f"Config {config_path} and model {ckpt} outputs saved to reconstructions_voxel_{name}.npz")

        # only do this once...
        if not voxels_truth_saved:
            voxels_truth = []
            for i in range(len(predictions)):
                for b in range(len(predictions[i][3])):    # batch
                    voxels_truth.append(predictions[i][3][b].cpu())
            np.savez(os.path.join(args.save_npz_dir,'voxels_truth.npz'), *voxels_truth)
            voxels_truth_saved = True
            print(f"voxels preprocessed truth saved to voxels_truth.npz")
            del voxels_truth
        
        del model,predictions,all_iou,all_codes,reconstructions
    
if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",     type=str, default="/mnt/data/Public_datasets/nuScenes/",    help="parent directory containing dataset files (gets joined with info file)")
    parser.add_argument("--batch-size",   type=int, default=12,             help="batch size")
    parser.add_argument("--num-workers",  type=int, default=4,              help="number of worker processes")
    parser.add_argument("--info-file",    type=str, default="data_managment/nuscenes_info.json", help="train/val/test data split is given by this file (gets joined with root_dir)")
    parser.add_argument("--save-npz-dir", type=str, default=os.path.join(root_dir, "logs"),           help="directory to save logs")
    parser.add_argument("--dataset-name", type=str, default="nuscenes",     help="either nuscenes or frontcam")
    parser.add_argument("--sigma",        type=float, default=0.3,     help="threshold for sigmoid (0.3 maximized IOU in our experiments)")
    
    parser.add_argument('--models', nargs='+', help='Models to evaluate', required=True)
    parser.add_argument('--names', nargs='+', help='Names of models to evaluate', required=True)
    parser.add_argument('--configs', nargs='+', help='Configs of models to evaluate', required=True)

    args = parser.parse_args()

    args.root_dir = root_dir
    args.info_file_path = os.path.join(root_dir, args.info_file)

    main(args)