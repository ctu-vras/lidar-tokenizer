conda create -n lidar_tokenizer python=3.10.11 -y
conda activate lidar_tokenizer
pip install --upgrade pip
pip install torchmetrics==0.5.0 pytorch-lightning==1.4.2 omegaconf==2.1.1 einops==0.3.0 transformers==4.36.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 opencv-python kornia==0.7.0 wandb more_itertools
conda install scipy
conda install matplotlib
pip install timm==0.5.4
pip install numpy==1.26.4
conda install ignite -c pytorch