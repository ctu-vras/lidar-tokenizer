# LiDAR Tokenizer
This is a repository containing code to train and evaluate two LiDAR tokenizer models presented in the diploma thesis "LiDAR Tokenizer" by Adam Herold 09/2024 - 01/2025. The two models are:
1) 2D range image model based on the auto-encoder from paper "*Towards Realistic Scene Generation with LiDAR Diffusion Models (Haoxi Ran, Vitor Guizilini , Yue Wang, 2024)*". A lot of the code in this repository has its source in their official repository at https://github.com/hancyran/LiDAR-Diffusion. 

2) 3D voxel model based on the auto-encoder from paper "*Learning Compact Representations for LiDAR Completion and Generation (Yuwen Xiong et al., 2023)*". Most of the code for this model has its source in the unofficial implementation at https://github.com/myc634/UltraLiDAR_nusc_waymo. 

Both models can be trained and evaluated on two datasets — nuScenes and FrontCam. It should be also easily extensible to KITTI.

## Contents
The implementation uses PyTorch Lightning.
There are two main sub-folders: 
 - `data_managment` contains classes handling data loading and data preprocessing (i.e. range projection or voxelization).
 - `model_managment` contains the code for the two models (coming mostly from their respective repositories). The implementation of the models' architectures is split into `encoders.py`, `decoders.py` and `codebooks.py`. In the latter there are the quantization algorithms used by the two models, but there is also FSQ adapted from the repository https://github.com/duchenzhuang/FSQ-pytorch/. The LightningModules are defined in `models.py`.

Besides that there are some `plotting_functions.py` that were used for the thesis. In the root directory there are the following scripts:
 - For evaluation: `evaluate_fpvd_cd.py`, `evaluate_range.py`, `eavluate_voxel.py` (on how to use these later).
 - `frontcam_calibrate.py` - FrontCam uses multiple LiDAR sensors with different coordinate frames, this was used to calibrate them together.
 - `frontcam_data_split.py` - Used to split into train/val/test datasets.
 - `frontcam_match_timestamps.py` - Used to match image-LiDAR file pairs.
 - `training.py` - Training of all models.

## Installing dependencies
If you run the commands from `env.sh` you should be able to run the training and evaluation commands. For completeness I have also included an `environment.yml` file of the working environment I used, but I think it is unnecessary bloated with unused packages.

## Training
Training of all models is done using `training.py` with proper arguments. Additionally one also has to specify a config file — the ones used in our experiments are located in `model_managment/final_configs`. The settings that were used in the thesis are the following:
#### nuScenes & 2D range image model 
 - `python training.py --config model_managment/final_configs/nuscenes/range_image/{ONE OF THE CONFIGS} --model-name {CHOOSE SOME NAME}`
#### nuScenes & 3D voxel model 
 - `python training.py --config model_managment/final_configs/nuscenes/voxel/{ONE OF THE CONFIGS} --batch-size 4 --model-name {CHOOSE SOME NAME} --max-steps 120000 --mode voxel`
#### FrontCam& 2D range image model 
 - `python training.py --config model_managment/final_configs/frontcam/range_image/{ONE OF THE CONFIGS} --model-name {CHOOSE SOME NAME} --dataroot {ROOT FILE OF DATA} --info-file data_managment/frontcam_all.json --dataset-name frontcam`
#### FrontCam& 3D voxel model 
 - `python training.py --config model_managment/final_configs/frontcam/voxel/{ONE OF THE CONFIGS} --batch-size 4 --model-name {CHOOSE SOME NAME} --max-steps 120000 --mode voxel --dataroot {ROOT FILE OF DATA} --info-file data_managment/frontcam_all.json --dataset-name frontcam`

I recommend checking the `training.py` file and making sure all of the other arguments' default values make sense for your setup. Some other options besides just training that are worth mentioning:
 - You can continue training by specifying `--resume-from-checkpoint {ABS PATH TO MODEL CKPT}.` Just be careful if you're using the OneCycle LR scheduler (which is used by default), that once you specify the number of steps you cannot (by default) run more steps afterwards.
 - You can look for optimal LR for OneCycle by adding `--auto-lr-find`. This worked well with the 2D range image model, but for the 3D voxel model I think it gave some weird value so I wouldn't rely on it too much.



## Evaluation
Evaluation is split into multiple files:
 - `evaluate_range.py` to evaluate range image models (either nuscenes or frontcam, not both at the same time...). This calculate the following metrics: number of points (in and out), MSE (range image and 3D points), MAE (range image and 3D points), SSIM, PSNR, codebook utilization and uniformity. Besides that it will also save .NPZ files of the ground_truth, of the preprocessed ground truth range_image_truth and of the output of each model (tip: these are saved into a directory given by `--save-npz-dir`). These are used to later calculate the FPVD and CD with another script and can also be used for visualizations and what not.
 Example nuScenes usage:
	- `python evaluate_range.py --models {ABS PATH MODEL 1} {ABS PATH MODEL 2} --names {SOME NAME 1} {SOME NAME 2} --configs model_managment/final_configs/nuscenes/range_image/{CONFIG FILE 1} model_managment/final_configs/nuscenes/range_image/{CONFIG FILE 2} --dataset-name nuscenes`

 - `evaluate_voxel.py` similar as previous but for voxel models. Calculates number of points, IOU and codebook metrics and also saves the .NPZ files. Usage on FrontCam:
	 - `python evaluate_voxel.py --models {ABS PATH MODEL 1} {ABS PATH MODEL 2} --names {SOME NAME 1} {SOME NAME 2} --configs model_managment/final_configs/frontcam/voxel/{CONFIG FILE 1} model_managment/final_configs/frontcam/voxel/{CONFIG FILE 2} --dataset-name frontcam --info-file data_managment/frontcam_decimated.json`

 - `evaluate_fpvd_cd.py` is used to calculate FPVD (nuscenes only) and CD (both datasets). It fully relies on the "evaluation toolbox" of the LiDAR Diffusion repository (https://github.com/hancyran/LiDAR-Diffusion/blob/main/lidm/eval/README.md).  To get it to work I did the following (warning: it is a bit clunky):
	 - `git clone https://github.com/hancyran/LiDAR-Diffusion`
	 - rename LiDAR-Diffusion to LD
	 - create `pretrained_weights` folder inside LD
	 - download pretrained models according to https://github.com/hancyran/LiDAR-Diffusion/blob/main/lidm/eval/README.md into `pretrained_weights` (so you should now have `LD/pretrained_weights/nuscenes/spvcnn/model.ckpt`)
	 - change `DEFAULT_ROOT` in `LD/lidm/eval/__init__.py` to the absolute path of `pretrained_weights`
	 - get the `evaluate_fpcd_cd.py` to run — the evaluation toolbox requires some specific libraries to run, I had some trouble getting it to work, what worked for me in the end was using a specific combination of installed modules on the cluster (you can see exactly what I did in `fpvd.sh`) 

	Example usage:
	`python evaluate_fpvd_cd.py --dataset-name nuscenes --load-npz-dir {WHEREVER YOU HAVE THE NPZ FILES SAVED} --range-image-files {NAME OF RANGE IMAGE FILE 1} {NAME OF RANGE IMAGE FILE 2} --voxel-files {NAME OF VOXEL FILE}`

## Final Remarks
This repository is a "cleaned-up" version of my working repository. All the example commands listed here I have tested and they should work, but I couldn't run everything fully, so please forgive if there is a hardcoded file name that throws an error somewhere :). Regrettably, I have not used a specific "random seed", so I would expect the reproduced results to not be completely 1:1 with the thesis. However, if anything seems off feel free to contact me, as I ran all of the experiments with the "working" repository and not this "cleaned-up" one.

ChatGPT has been used at times as a helping tool for writing code (mostly plotting functions).

Finally, let me once again list all the other repositories that were used to make this one:
 - 2D range image model; CD and FPVD evaluation: https://github.com/hancyran/LiDAR-Diffusion
 - 3D voxel model: https://github.com/myc634/UltraLiDAR_nusc_waymo
 - FSQ: https://github.com/duchenzhuang/FSQ-pytorch
 - Nuscenes Dataset class: https://github.com/valeoai/rangevit
