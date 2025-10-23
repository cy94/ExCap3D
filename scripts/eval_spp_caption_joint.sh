#!/bin/bash

#SBATCH --job-name=m3d_cap_joint_eval
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|rtx_a6000"

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_cap_joint_eval_%j.log
#SBATCH --partition=submit  
#SBATCH --exclude=daidalos

eval "$(conda shell.bash hook)"
conda activate mask3d_2

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -ex
echo "Running on" `uname -n`

python main_instance_segmentation.py \
    general.train_mode=false \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance"  \
    data.data_dir=/cluster/eriador/cyeshwanth/caption3d/mask3d/spp_data_0.005_40_v2_alpha \
    data.semantic_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/semantic_benchmark/top100.txt \
    data.instance_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/top17_instance_from_100_v2.txt \
    general.eval_on_train=false \
    general.gen_captions=true general.obj_caption_key=summarized_local_caption_16 \
    general.gen_part_captions=true general.part_caption_key=summarized_parts_caption_16 \
    data.caption_data_dir=/cluster/eriador/cyeshwanth/caption3d/gen_captions/sub10_local_gen5_part/scene_captions_84 \
    data.instance_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/semantic_benchmark/top100_instance.txt caption_model.class_weights_file=null \
    'general.wandb_group="evaluate"' \
    'general.notes="evaluate full model"' \
    'general.checkpoint="/cluster/eriador/cyeshwanth/caption3d/mask3d/checkpoints/2350484/epoch=68val_cider=0.367.ckpt"' \
    data.validation_dataset.list_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/splits/nvs_sem_val.txt \
    general.obj_output_to_part=false \
    general.part_output_to_obj=true \
    caption_model.use_obj_segment_feats=true \
    general.project_hidden_states=false \
    general.use_hidden_state_ndx=1 \