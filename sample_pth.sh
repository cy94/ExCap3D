#!/bin/bash

#SBATCH --job-name=m3d_sample_pth 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=128gb                     
#SBATCH --cpus-per-task=8

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_sample_pth_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"
conda activate mask3d_2

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -ex

python run_seg_parallel.py preprocess \
    --data_root=/cluster/eriador/cyeshwanth/scannetpp_download_v2/data/ \
    --segmentThresh=0.005 \
    --list_file=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_train_val.txt \
    --segmentMinVertex=40 \
    --out_dir=/cluster/eriador/cyeshwanth/caption3d/segdata/segments_0.005_40_v2/ \

# create pth files with on sampled points - can be used for training anything
python sample_pth.py \
    n_jobs=8 \
    data_dir=/cluster/eriador/cyeshwanth/scannetpp_download_v2/data/ \
    input_pth_dir=/cluster/eriador/cyeshwanth/caption3d/pth_data_100_v2 \
    list_path=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_train_val.txt \
    segments_dir=/cluster/eriador/cyeshwanth/caption3d/segdata/segments_0.005_40_v2/ \
    output_pth_dir=/cluster/eriador/cyeshwanth/caption3d/pth_data/pth_data_100_0.1_0.005_40_v2 \
    sample_factor=0.1 \

# prepare data in mask3d format - npy, with database files, etc
python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=8 \
    --data_dir=/cluster/eriador/cyeshwanth/caption3d/pth_data/pth_data_100_0.1_0.005_40_v2 \
    --save_dir=/cluster/eriador/cyeshwanth/caption3d/mask3d/spp_data_0.005_40_v2 \
    --train_list=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_train.txt \
    --val_list=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_val.txt \

