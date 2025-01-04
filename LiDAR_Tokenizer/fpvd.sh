#!/bin/bash -l

### This is an example on how I ran the FPVD and CD metric evaluation on our cluster.

#SBATCH --job-name=metrics
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB   


ml purge

SCRATCH_DIRECTORY=/data/temporary/${USER}/${SLURM_JOBID}.stallo-adm.uit.no
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

cp -r /home/${USER}/lidar-tokenizer-final/LiDAR-Tokenizer/fid_evaluation/nuscenes/* ${SCRATCH_DIRECTORY} 

ml geotransformer/1.0.0
ml PyTorch-Lightning/1.6.5-foss-2021a-CUDA-11.3.1
ml foss/2021a Python/3.9.5-GCCcore-10.3.0 sparsehash/2.0.4-GCCcore-10.3.0 PyTorch/1.10.0-foss-2021a-CUDA-11.3.1 spconv/2.1.21-foss-2021a-CUDA-11.3.1 CUDA/11.3.1
ml torchsparse/1.4.0-foss-2021a-CUDA-11.3.1

python /home/${USER}/lidar-tokenizer-final/LiDAR-Tokenizer/evaluate_fpvd_cd.py --range-image-files reconstructions_range_image_fsq_4k_1.npz reconstructions_range_image_fsq_4k_2.npz

ml purge

cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}

exit 0