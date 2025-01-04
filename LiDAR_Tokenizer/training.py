# %%
import importlib
#importlib.reload(my_module)

from data_managment import datasets, nuscenes
from data_managment.data_modules import LidarTokenizerModule
from model_managment.models import LidarDiffusionModel
from model_managment.helpers import load_model

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import os
import sys
from datetime import datetime
import argparse


def main(args):
    data = LidarTokenizerModule(dataroot = args.dataroot,
                      batch_size = args.batch_size,
                      num_workers = args.num_workers,
                      info_file_path = args.info_file_path,    # specifies which files are train/val/test
                      root_dir = args.root_dir,
                      mode = args.mode,
                      dataset = args.dataset_name,
                      config_path = args.config_path
                      )   
    
    model = load_model(args.config_path)

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    log_fn = f"{args.model_name}_{timestamp}"
    tb_logger = TensorBoardLogger(args.log_dir, name=log_fn)

    checkpoint_callback = ModelCheckpoint(
        monitor = None,     
        dirpath = args.ckpt_log_dir,
        verbose=True,
        every_n_epochs = 1,
        save_top_k=-1,
        save_on_train_epoch_end = True,
        filename = log_fn+'-{epoch:02d}-{step}'
    )

    checkpoint_callback_best = ModelCheckpoint(
        monitor = 'val/points_mse',
        dirpath = args.ckpt_log_dir,
        every_n_epochs = 1,
        filename = log_fn+'-best-{epoch:02d}-{step}'
        )

    if args.auto_lr_find:
        trainer = pl.Trainer(
            auto_lr_find=True
        )
        trainer.tune(model, data)

    else:
        trainer = pl.Trainer(
            logger = tb_logger,
            max_steps = args.max_steps,
            gpus = 1,
            callbacks = [checkpoint_callback, checkpoint_callback_best],
            accumulate_grad_batches = args.grad_cum,
            gradient_clip_val = args.grad_clip,
            resume_from_checkpoint = args.resume_from_checkpoint
        )

        trainer.fit(model, data)
    

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="model_managment/final_configs/config_FINAL_ld_fsq_4k.yaml", help="config file (gets joined with root_dir)")
    parser.add_argument("--dataroot",     type=str, default="/mnt/data/Public_datasets/nuScenes/",    help="parent directory containing dataset files (gets joined with info file)")
    parser.add_argument("--batch-size",   type=int, default=12,             help="batch size")
    parser.add_argument("--num-workers",  type=int, default=4,              help="number of worker processes")
    parser.add_argument("--info-file",    type=str, default="data_managment/nuscenes_info.json", help="train/val/test data split is given by this file (gets joined with root_dir)")
    parser.add_argument("--model-name",   type=str, default="range_image_nuscenes",                  help="name under which to save model")
    parser.add_argument("--ckpt-log-dir", type=str, default=os.path.join(root_dir, "saved_models"),   help="directory to save trained models")
    parser.add_argument("--log-dir",      type=str, default=os.path.join(root_dir, "logs"),           help="directory to save logs")
    parser.add_argument("--max-steps",    type=int, default=80000,          help="number of training steps (used instead of number of epochs)")
    parser.add_argument("--grad-cum",     type=int, default=2,              help="gradient accumulation")
    parser.add_argument("--grad-clip",    type=int, default=5,              help="gradient clipping")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="path to model you want to resume training")
    parser.add_argument("--mode",         type=str, default="range_image",  help="either range_image or voxel")
    parser.add_argument("--dataset-name", type=str, default="nuscenes",     help="either nuscenes or frontcam")
    parser.add_argument("--auto-lr-find", action='store_true',        help="when you don't want to train but want to search for best LR (using lightning)")

    args = parser.parse_args()

    args.config_path = os.path.join(root_dir, args.config)
    args.info_file_path = os.path.join(root_dir, args.info_file)

    args.root_dir = root_dir
    
    main(args)