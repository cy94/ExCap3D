#!/bin/bash

#SBATCH --job-name=spp_preprocess 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --cpus-per-task=8

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/spp_preprocess_%j.log
#SBATCH --partition=submit  

python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --data_dir=/cluster/eriador/cyeshwanth/caption3d/pth_data_100_0.1_0.005_320 \
    --save_dir=/cluster/eriador/cyeshwanth/caption3d/mask3d/spp_data_0.005_320 \
    --train_list=/cluster/eriador/cyeshwanth/caption3d/spp_meta/split_public/train_25.txt \
    --val_list=/cluster/eriador/cyeshwanth/caption3d/spp_meta/split_public/train_25.txt \