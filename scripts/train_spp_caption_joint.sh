#!/bin/bash

#SBATCH --job-name=m3d_cap_joint 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|rtx_a6000"

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_cap_joint_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"
conda activate mask3d_2

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -ex
echo "Running on" `uname -n`

python main_instance_segmentation.py \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance"  \
    data.data_dir=/cluster/eriador/cyeshwanth/caption3d/mask3d/spp_data_0.005_40_v2_alpha \
    data.train_dataset.list_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/splits/nvs_sem_train.txt \
    data.validation_dataset.list_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/splits/nvs_sem_val.txt \
    data.instance_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/semantic_benchmark/top100_instance.txt caption_model.class_weights_file=null \
    data.semantic_classes_file=/cluster/eriador/cyeshwanth/scannetpp_download_v2_alpha/metadata/semantic_benchmark/top100.txt \
    general.freeze_segmentor=true \
    general.gen_captions=true general.obj_caption_key=summarized_local_caption_16 \
    general.gen_part_captions=true general.part_caption_key=summarized_parts_caption_16 \
    data.caption_data_dir=/cluster/eriador/cyeshwanth/caption3d/gen_captions/sub10_local_gen5_part/scene_captions \
    scheduler=cosine optimizer.lr=5e-3 \
    'general.checkpoint="/cluster/eriador/cyeshwanth/caption3d/mask3d/checkpoints/2183170-instseg-84cls-v2a/epoch=229-val_ap50_val_mean_ap_50=0.387.ckpt"' \
    "general.wandb_group="train"" 
    'general.notes="train full model"' \
    general.obj_output_to_part=false \
    general.part_output_to_obj=true \
    caption_model.use_obj_segment_feats=true \
    general.project_hidden_states=false \
    general.use_hidden_state_ndx=1 \
    consistency=base consistency.classification=true \
    consistency.classification_model.aggr=mean \
    consistency.loss_weights.classification=0.1 \
    consistency=base consistency.similarity=true \
    consistency.similarity_type=cosine \
    consistency.projection_model.aggr=max \
    consistency.loss_weights.similarity=0.1 \