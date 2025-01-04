import sys
sys.path.insert(0,'LiDAR_Tokenizer/LD')
from LD.lidm.eval.eval_utils import evaluate
import numpy as np
import os
import sys
import time
import argparse

def main(args):
    data = '32'  # specify data type to evaluate

    if args.dataset_name == "nuscenes":
        metrics = ['cd','fpvd']  # specify metrics to evaluate
        print("Computing CD and FPVD")
    elif args.dataset_name == "frontcam":
        metrics = ['cd']         # can't compute fpvd because there is no pretrained model on this dataset
        print("Computing CD only for FrontCam")
    else:
        raise NotImplementedError(f"invalid dataset_name {args.dataset_name}. choose one of 'nuscenes' or 'frontcam'")
    

    range_image_files = args.range_image_files
    voxel_files = args.voxel_files
                
    ground_truth = np.load(os.path.join(args.load_npz_dir,'ground_truth.npz'))
    ground_truth = np.array([ground_truth[f'arr_{i}'].astype(np.float32) for i in range(len(ground_truth))], dtype=object)

    if range_image_files:
        # range image
        range_image_truth = np.load(os.path.join(args.load_npz_dir,'range_image_truth.npz'))
        range_image_truth = np.array([range_image_truth[f'arr_{i}'].astype(np.float32) for i in range(len(range_image_truth))], dtype=object)

        for f in range_image_files:
            reconstructions = np.load(os.path.join(args.load_npz_dir,f))
            reconstructions = np.array([reconstructions[f'arr_{i}'].astype(np.float32) for i in range(len(reconstructions))], dtype=object)
            print(f"ground truth & {f}")
            evaluate(ground_truth, reconstructions, metrics, data)
            print(f"range_image truth & {f}")
            evaluate(range_image_truth, reconstructions, metrics, data)

        print("range_image truth & ground truth")
        evaluate(ground_truth, range_image_truth, metrics, data)
        del range_image_truth

    if voxel_files:
        # voxel
        voxels_truth = np.load(os.path.join(args.load_npz_dir,'voxels_truth.npz'))
        voxels_truth = np.array([voxels_truth[f'arr_{i}'].astype(np.float32) for i in range(len(voxels_truth))], dtype=object)

        for f in voxel_files:
            reconstructions = np.load(os.path.join(args.load_npz_dir,f))
            reconstructions = np.array([reconstructions[f'arr_{i}'].astype(np.float32) for i in range(len(reconstructions))], dtype=object)
            print(f"ground truth & {f}")
            evaluate(ground_truth, reconstructions, metrics, data)
            print(f"voxels truth & {f}")
            evaluate(voxels_truth, reconstructions, metrics, data)

        print("voxels truth & ground truth")
        evaluate(ground_truth, voxels_truth, metrics, data)



if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-npz-dir", type=str, default=os.path.join(root_dir, "logs"),           help="directory to save logs")
    parser.add_argument("--dataset-name", type=str, default="nuscenes", help="either nuscenes or frontcam")
    
    parser.add_argument('--range-image-files', nargs='+', help='Models to evaluate', required=False)
    parser.add_argument('--voxel-files', nargs='+', help='Names of models to evaluate', required=False)

    args = parser.parse_args()

    main(args)