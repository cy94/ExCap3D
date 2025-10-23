#!/bin/bash

#SBATCH --job-name=m3d_spp_instseg     
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="rtx_a6000"

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_spp_instseg_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"
conda activate mask3d_2

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# print node name using hostname
echo "Running on $(hostname)"

python main_instance_segmentation.py \
    general.save_root=/cluster/eriador/cyeshwanth/caption3d/mask3d/checkpoints \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.train_dataset.clip_points=300000 \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance"  \
    data.data_dir=/cluster/eriador/cyeshwanth/caption3d/mask3d/spp_data_0.005_40_v2_alpha \
    data.train_dataset.list_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/splits/nvs_sem_train.txt \
    data.validation_dataset.list_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/splits/nvs_sem_val.txt \
    data.semantic_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2/metadata/semantic_benchmark/top100.txt \
    data.instance_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/semantic_benchmark/top100_instance.txt caption_model.class_weights_file=null \
    data.batch_size=6 \
    'general.wandb_group="train instance segmentation"' \
    'general.notes="train"' \
    

