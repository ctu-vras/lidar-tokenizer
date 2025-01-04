# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group

from Trainer.vit import MaskGIT


def main(args):
    print(f"Running on device {args.device}, rank {os.environ['LOCAL_RANK']}")
    """ Main function:Train or eval MaskGIT """

    #assert args.mask_value == args.codebook_size_img + args.codebook_size_lidar 
    maskgit = MaskGIT(args)

    if args.pcd2img:
        maskgit.conditioned(
                        sm_temp=args.sm_temp,
                        randomize="linear",
                        r_temp=args.r_temp,
                        sched_mode=args.sched_mode,
                        step=args.step,
                        img2pcd = False,
                        k=100)
    elif args.img2pcd:
        maskgit.conditioned(
                        sm_temp=args.sm_temp,
                        randomize="linear",
                        r_temp=args.r_temp,
                        sched_mode=args.sched_mode,
                        step=args.step,
                        img2pcd = True,
                        k=100)
    elif args.plot_imgs:
        maskgit.plot_images()
    elif args.plot_img_range_img:
        maskgit.plot_img_range_img()  
    elif args.plot_pcd:
        maskgit.plot_pcd()
    elif args.sample_images:  # Evaluate the networks
        maskgit.generate_samples()

    elif args.test_only:  # Evaluate the networks
        maskgit.eval()

    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)] * 1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            sm_temp = 1.3          # Softmax Temperature
            r_temp = 7             # Gumbel Temperature
            w = 9                  # Classifier Free Guidance
            randomize = "linear"   # Noise scheduler
            step = 32              # Number of step
            sched_mode = "arccos"  # Mode of the scheduler
            # Generate sample
            gen_sample, _, _ = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                              randomize=randomize, sched_mode=sched_mode, step=step)
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            save_image(gen_sample, f"saved_img/sched_{sched_mode}_step={step}_temp={sm_temp}"
                                   f"_w={w}_randomize={randomize}_{name}.jpg")
    else:  # Begin training
        maskgit.fit()


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   default="frontcam", help="dataset on which dataset to train")
    parser.add_argument("--data-info",         type=str,   default="Dataset/frontcam_tokens.json", help="splits' file paths")
    parser.add_argument("--data-folder",  type=str,   default="/data",         help="folder containing the dataset")
    parser.add_argument("--vqgan-folder", type=str,   default="LiDAR_MaskGIT/pretrained_maskgit/VQGAN/",         help="folder of the pretrained VQGAN")
    parser.add_argument("--vit-folder",   type=str,   default="LiDAR_MaskGIT/logs/1/checkpoints/",         help="folder where to save the Transformer")
    parser.add_argument("--writer-log",   type=str,   default="LiDAR_MaskGIT/logs/1/tb_logs/",         help="folder where to store the logs")
    parser.add_argument("--sched_mode",   type=str,   default="arccos",   help="scheduler mode whent sampling")
    parser.add_argument("--grad-cum",     type=int,   default=2,          help="accumulate gradient")
    parser.add_argument('--channel',      type=int,   default=3,          help="rgb or black/white image")
    parser.add_argument("--num_workers",  type=int,   default=2,          help="number of workers")
    parser.add_argument("--step",         type=int,   default=15,          help="number of step for sampling")
    parser.add_argument('--seed',         type=int,   default=42,         help="fix seed")
    parser.add_argument("--epoch",        type=int,   default=80,        help="number of epoch")
    parser.add_argument('--img-size',     type=int,   default=512,        help="image size")
    parser.add_argument("--bsize",        type=int,   default=2,        help="batch size")
    #parser.add_argument("--mask-value",   type=int,   default=5399,       help="") # instead the codebook size is used for the mask value...
    parser.add_argument("--lr",           type=float, default=1e-4,       help="learning rate to train the transformer")
    parser.add_argument("--cfg_w",        type=float, default=3,          help="classifier free guidance wight")
    parser.add_argument("--r_temp",       type=float, default=7.,        help="Gumbel noise temperature when sampling")
    parser.add_argument("--sm_temp",      type=float, default=1.,         help="temperature before softmax when sampling")
    parser.add_argument("--drop-label",   type=float, default=0.1,        help="drop rate for cfg")
    parser.add_argument("--test-only",    action='store_true',            help="only evaluate the model")
    parser.add_argument("--save-samples-folder",    type=str, default="/data/sampled_images/",    help="")
    parser.add_argument("--resume",       action='store_true',            help="resume training of the model")
    parser.add_argument("--debug",        action='store_true',            help="debug")
    #parser.add_argument("--num-tokens-img",        type=int, default=1024,            help="") # is currently calculated in MaskGIT as patch_size^2
    parser.add_argument("--num-tokens-lidar",      type=int, default=1008,            help="number of tokens one lidar is encoded as")  #1008
    parser.add_argument("--num-tokens-lidar-h",      type=int, default=24,            help="for range image its the dimensions of the range image and for voxel dimensions of the grid")
    parser.add_argument("--num-tokens-lidar-w",      type=int, default=42,            help="for range image its the dimensions of the range image and for voxel dimensions of the grid")
    parser.add_argument("--codebook-size-img",      type=int, default=1024,            help="number of possible codes in the the codebook of the image tokenizer")
    parser.add_argument("--codebook-size-lidar",      type=int, default=4375,            help="number of possible codes in the the codebook of the lidar tokenizer") #4375
    parser.add_argument("--lidar-config", type=str, default="LiDAR_MaskGIT/LiDAR_Tokenizer/model_managment/final_configs/frontcam/range_image/config_FINAL_FRONTCAM_ld_fsq.yaml", help="lidar tokenizer config")
    parser.add_argument("--lidar-ckpt", type=str, default="LiDAR_MaskGIT/LiDAR_Tokenizer/model_managment/saved_models/range_model.ckpt", help="lidar tokenizer model")
    parser.add_argument("--img-only", action='store_true', help="FLAG TO ONLY USE IMAGE MODALITY AND NO LIDAR MODALITY (one of the three settings in the thesis)")
    parser.add_argument("--sample-images", action='store_true', help="generate sampled images (the number of which is hardcoded in vit.py to be 19400 to match the number of testing images)")
    parser.add_argument("--onecycle", action='store_true', help="to activate onecycle LR scheduler (default is just constant LR)")
    parser.add_argument("--pcd2img", action='store_true', help="generate lidar conditioned images")
    parser.add_argument("--img2pcd", action='store_true', help="generate image conditioned lidars")
    parser.add_argument("--plot-imgs", action='store_true', help="plot sampled images")
    parser.add_argument("--plot-img-range-img", action='store_true', help="plot sampled images together with corresponding range images")
    parser.add_argument("--plot-pcd", action='store_true', help="plot sampled images and create corresponding npz files with point clouds")
    parser.add_argument("--mode", type=str, default="range_image" , help="either range_image or voxel")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0

    print(f"IMG ONLY: {args.img_only}")
    print(f"ONECYCLE: {args.onecycle}")

    if args.seed > 0: # Set the seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()
    args.num_gpus = world_size

    if world_size > 1:  # launch multi training
        print(f"{world_size} GPU(s) found, launch multi-gpus training")
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:  # launch single Gpu training
        print(f"{world_size} GPU found")
        args.is_master = True
        args.is_multi_gpus = False
        main(args)
