# LiDAR Tokenizer — MaskGIT
This is a repository containing code to train and evaluate a MaskGIT model adjusted for two modalities — image and LiDAR. It is related to the diploma thesis "LiDAR Tokenizer" by Adam Herold 09/2024 - 01/2025. This is only a secondary part of the thesis, the main part concerning the LiDAR tokenizers is located at TODO.

This repository expands on the Pytorch implementation of MaskGIT at https://github.com/valeoai/Maskgit-pytorch by adding the LiDAR modality.
The original paper on MaskGIT is *MaskGIT: Masked Generative Image Transformer. Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman, 2022.*
The Pytorch implementation paper is *A Pytorch Reproduction of Masked Generative Image Transformer. Victor Besnier, Mickael Chen, 2023.*
So also read the instructions there, e.g. for what image tokenizer to get.

Following is the description of the adjustments made to the original repository to allow for the usage of the LiDAR modality.

## LiDAR modality
First of all, even though I tried to keep this package separate from the `LiDAR_Tokenizer` package, it is necessary to run some things in here. So I recommend to first clone the `LiDAR_Tokenizer` into the root of this package. Of course you also need a pretrained LiDAR Tokenizer model (one of the two provided by the aforementioned package, that is 2D range image or 3D voxel tokenizer) and its config (in this config you might need to prepend the name of the package to the relative paths e.g. `model_managment.something`->`LiDAR_Tokenizer.model_managment.something`).

We keep the same image tokenizer (ImageNet pre-trained VQ-GAN) and we add our own two LiDAR tokenizers (2D range image and 3D voxel LiDAR tokenizers). We decide to pre-compute all tokens for all data samples, store them and load them while training, instead of tokenizing dynamically while running. For this run the script `extract_img_lidar_tokens.py` (make sure to specify all the arguments, which are the locations of the trained tokenizers and folder which contains the data). The script assumes you are using the FrontCam dataset in the same form as I used it, if that's not the case, adjust the names of the `img_dirs` and `lidar_dirs`. Besides, you can use the `frontcam_data_split.py` to obtain a json file specifying which files (the tokens but also the original images and pointclouds, which are used for visualization) are used for training and which for testing.

Once you have the tokens you can run the training of the MaskGIT. The MaskGIT was changed to take both modalities, how exactly they are treated is described in the thesis, but the short version is the tokens are transformed into the underlying embedding vectors, the position embedding is added and then the two modalities are concatenated and processed together by the transformers. On output they are split again... 

## Installing dependencies
A different environment is used from the `LiDAR_Tokenizer`! I use the `environment.yml` from the original repository but I added `plotly` and `timm==0.5.4`, I think that should be all the necessary additions.

## Training
Training is done using `main.py` with proper arguments. There are three setups that have been presented in the thesis:
#### Image only
`torchrun  --nproc_per_node={NUM GPUs}  main.py  --onecycle --img-only `
#### Image with range image LiDAR
`torchrun  --nproc_per_node={NUM GPUs} main.py  --onecycle --mode range_image --data-info Dataset/frontcam_tokens.json --lidar-config  {LIDAR_TOKENIZER_CONFIG_PATH} --lidar-ckpt  {LIDAR_TOKENIZER_CKPT_PATH} `
#### Image with voxel LiDAR
`torchrun  --nproc_per_node={NUM GPUs} main.py  --onecycle --mode voxel --data-info Dataset/frontcam_tokens_voxel.json --num-tokens-lidar 960 --num-tokens-lidar-h 24 --num-tokens-lidar-w 40 --codebook-size-lida 4098 --lidar-config  {LIDAR_TOKENIZER_CONFIG_PATH} --lidar-ckpt  {LIDAR_TOKENIZER_CKPT_PATH}`

I recommend to check the `main.py` file and all the arguments inside. In all three cases you will also have to specify the following folder paths:
`--data-folder {...} --vqgan-folder {...} --vit-folder {...} --writer-log {...}`

## Evaluation
For the FID I sythesized 19364 images to match the test set number of images:
 - `torchrun --nproc_per_node=1 main.py --sample-images --resume --bsize 16 --vit-folder {TRAINED_MASKGIT_PATH} --step 15 --r_temp 7 --test-only`

Then I used the following: 
 - **FID** — I used the pytorch_fid package https://github.com/mseitzer/pytorch-fid 
 - **Precision, Recall (F1/8 and F8)** — I used this package https://github.com/msmsajjadi/precision-recall-distributions/ which is the official repository for the paper M. S. M. Sajjadi, O. Bachem, M. Lucic, O. Bousquet, and S. Gelly, “Assessing generative models via precision and recall”, NeurIPS 2018.

Alternatively, one can use the `main.py --test-only` which was used in the original MaskGIT-pytorch repository.
 
 The inference (generating images, image to lidar, etc.) is done using `main.py`:

 -  Plot images `torchrun --nproc_per_node=1 main.py --plot-imgs --resume --bsize 16 --vit-folder {TRAINED_MASKGIT_PATH} --seed 1234561 --step 15 --r_temp 7`
 - LiDAR conditioned image: `torchrun --nproc_per_node=1 main.py --pcd2img --resume --bsize 3 --vit-folder {TRAINED_MASKGIT_PATH} --seed 1234561 --step 15 --r_temp 7`
 - Image conditioned LiDAR `torchrun --nproc_per_node=1 main.py --img2pcd --resume --bsize 3 --vit-folder {TRAINED_MASKGIT_PATH} --seed 1234561 --step 15 --r_temp 7`
 - Plot images and corresponding range images `torchrun --nproc_per_node=1 main.py --plot-img-range-img --resume --bsize 3 --vit-folder {TRAINED_MASKGIT_PATH} --seed 1234561 --step 15 --r_temp 7`
 - Plot images and save corresponding point clouds as .NPZ file (this time an example with voxel tokenizer) `torchrun --nproc_per_node=1 main.py --plot-pcd --resume --bsize 1 --vit-folder {TRAINED_MASKGIT_PATH} --seed 1234562 --step 15 --r_temp 7 --mode voxel --data-info Dataset/frontcam_tokens_voxel.json --num-tokens-lidar 960 --num-tokens-lidar-h 24 --num-tokens-lidar-w 40 --codebook-size-lidar 4098 --lidar-config {LIDAR_TOKENIZER_CONFIG_PATH} --lidar-ckpt {LIDAR_TOKENIZER_CHECKPOINT_PATH}`
 The .NPZ file with the point cloud can then be visualized using `python plots.py` and specifying the file.

## Final Remarks
This repository is a "cleaned-up" version of my working repository. All the commands listed here I have tested to work. However, if anything seems off feel free to contact me, as I ran all of the experiments with the "working" repository and not this "cleaned-up" one.

ChatGPT has been used at times as a helping tool for writing code (mostly plotting functions).

Finally, let me once again state this repository was made by modifying this one:
 - MaskGIT Pytorch imlpementation: https://github.com/valeoai/Maskgit-pytorch

And for evaluation I used:
 - FID: https://github.com/mseitzer/pytorch-fid
 - Precision & Recall: https://github.com/msmsajjadi/precision-recall-distributions