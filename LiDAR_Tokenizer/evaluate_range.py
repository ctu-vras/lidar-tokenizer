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
    
    range_image_truth_saved = False

    for k in range(len(configs)):
        name = names[k]
        config_path = config_paths[k]
        ckpt = models[k]

        model = load_model(config_path)
        checkpoint = torch.load(ckpt)#, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        data = LidarTokenizerModule(dataroot = args.dataroot,
                      batch_size = args.batch_size,
                      num_workers = args.num_workers,
                      info_file_path = args.info_file_path,    # specifies which files are train/val/test
                      root_dir = args.root_dir,
                      mode = "range_image",
                      dataset = args.dataset_name,
                      config_path = config_path,
                      version="v1.0-trainval"
                      ) 

        trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=0)
        predictions = trainer.predict(model, datamodule=data)
        num_points_in = 0
        num_points_out = 0

        test_img_mse_all = []
        test_img_mae_all = []
        test_points_mse_all = []
        test_points_mae_all = []
        ssim_all = []
        psnr_all = []

        def eval_step(engine, batch):
            return batch

        metric = SSIM(data_range=1.0)
        default_evaluator = Engine(eval_step)
        metric.attach(default_evaluator, 'ssim')

        metric_2 = PSNR(data_range=1.0)
        default_evaluator_2 = Engine(eval_step)
        metric_2.attach(default_evaluator_2, 'psnr')

        all_codes = torch.Tensor()
        all_codes = all_codes.int()

        for i in range(len(predictions)):
            for j in range(len(predictions[i]['test_img_mse'])):
                # MSE & MAE
                test_img_mse_all.append(predictions[i]['test_img_mse'][j])
                test_img_mae_all.append(predictions[i]['test_img_mae'][j])
                test_points_mse_all.append(predictions[i]['test_points_mse'][j])
                test_points_mae_all.append(predictions[i]['test_points_mae'][j])

                # SSIM & PSNR
                x_rec_masked = torch.from_numpy(predictions[i]['x_rec'][j])
                m_rec = predictions[i]['m_rec'][j]
                x_rec_masked[torch.argwhere(m_rec<0)[:,0], torch.argwhere(m_rec<0)[:,1]] = 0
                x_rec_masked = torch.unsqueeze(x_rec_masked,0)
                x_rec_masked = torch.unsqueeze(x_rec_masked,0)

                x_masked = torch.from_numpy(predictions[i]['x'][j])
                m = predictions[i]['m'][j]
                x_masked[torch.argwhere(m<0)[:,0], torch.argwhere(m<0)[:,1]] = 0
                x_masked = torch.unsqueeze(x_masked,0)
                x_masked = torch.unsqueeze(x_masked,0)

                min_val = torch.min(x_masked)
                max_val = torch.max(x_masked)
                
                x_rec_masked = (x_rec_masked-min_val)/(max_val-min_val)
                x_rec_masked = torch.clamp(x_rec_masked, 0., 1.)
                x_masked = (x_masked-min_val)/(max_val-min_val)

                state_ssim = default_evaluator.run([[x_rec_masked,
                                                x_masked]])
                state_psnr = default_evaluator_2.run([[x_rec_masked,
                                                x_masked]])
                
                ssim = state_ssim.metrics['ssim']
                ssim_all += [ssim]
                
                psnr = state_psnr.metrics['psnr']
                psnr_all += [psnr]

                # CODEBOOK UTILIZATION & UNIFORMITY
                all_codes = torch.cat((all_codes, predictions[i]['ind'][j].cpu()))
                num_points_in += predictions[i]['pcd'][j].shape[0]
                num_points_out += predictions[i]['pcd_rec_masked'][j].shape[0]

        print(f"num points in mean: {num_points_in/(len(predictions)*args.batch_size)}")
        print(f"num points out mean: {num_points_out/(len(predictions)*args.batch_size)}")
        top_k = 10
        print(f"model: {ckpt}")
        print(f"config: {config_path}")
        print(f" Range img MSE: mean = {round(np.mean(test_img_mse_all), 3):.3f}, std = {round(np.std(test_img_mse_all), 3):.3f}")
        print(f" Range img MAE: mean = {round(np.mean(test_img_mae_all), 3):.3f}, std = {round(np.std(test_img_mae_all), 3):.3f}")
        print(f"Pointcloud MSE: mean = {round(np.mean(test_points_mse_all), 3):.3f}, std = {round(np.std(test_points_mse_all), 3):.3f}")
        print(f"Pointcloud MAE: mean = {round(np.mean(test_points_mae_all), 3):.3f}, std = {round(np.std(test_points_mae_all), 3):.3f}")
        print(f"SSIM: mean = {round(np.mean(ssim_all), 3):.3f}, std = {round(np.std(ssim_all), 3):.3f}")
        print(f"PSNR: mean = {round(np.mean(psnr_all), 3):.3f}, std = {round(np.std(psnr_all), 3):.3f}")
        print(f"Codebook utilization on test dataset of {len(predictions)*args.batch_size} samples and {model.n_embed} codes: {len(all_codes.unique())/model.n_embed}")
        print(f"Codebook uniformity on test dataset of {len(predictions)*args.batch_size} samples: {all_codes.bincount().sort().values[-top_k:].sum()/len(all_codes)}")
        
        del model,test_img_mse_all,test_img_mae_all,test_points_mse_all,test_points_mae_all,ssim_all,psnr_all,all_codes
        del metric,metric_2,default_evaluator,default_evaluator_2

        reconstructions = []

        for i in range(len(predictions)):
            for j in range(len(predictions[i]['pcd_rec_masked'])):
                reconstructions.append(predictions[i]["pcd_rec_masked"][j])

        np.savez(os.path.join(args.save_npz_dir,f'reconstructions_range_image_{name}.npz'), *reconstructions)
        print(f"Config {config_path} and model {ckpt} outputs saved to reconstructions_range_image_{name}.npz")

        # only do this once...
        if not range_image_truth_saved:
            range_image_truth = []
            for i in range(len(predictions)):
                for j in range(len(predictions[i]['pcd'])):
                    range_image_truth.append(predictions[i]["pcd"][j])
            np.savez(os.path.join(args.save_npz_dir,'range_image_truth.npz'), *range_image_truth)
            range_image_truth_saved = True
            print(f"range_image preprocessed truth saved to range_image_truth.npz")
            del range_image_truth
        
        del predictions,reconstructions
    
if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",     type=str, default="/mnt/data/Public_datasets/nuScenes/",    help="parent directory containing dataset files (gets joined with info file)")
    parser.add_argument("--batch-size",   type=int, default=12,             help="batch size")
    parser.add_argument("--num-workers",  type=int, default=4,              help="number of worker processes")
    parser.add_argument("--info-file",    type=str, default="data_managment/nuscenes_info.json", help="train/val/test data split is given by this file (gets joined with root_dir)")
    parser.add_argument("--save-npz-dir",      type=str, default=os.path.join(root_dir, "logs"),           help="directory to save logs")
    parser.add_argument("--dataset-name", type=str, default="nuscenes",     help="either nuscenes or frontcam")
    
    parser.add_argument('--models', nargs='+', help='Models to evaluate', required=True)
    parser.add_argument('--names', nargs='+', help='Names of models to evaluate', required=True)
    parser.add_argument('--configs', nargs='+', help='Configs of models to evaluate', required=True)

    args = parser.parse_args()

    args.root_dir = root_dir
    args.info_file_path = os.path.join(root_dir, args.info_file)

    main(args)