from tqdm import tqdm
from copy import deepcopy
import gc
from contextlib import nullcontext
from utils.votenet_utils.metric_util import calc_iou
import json
from pathlib import Path
import statistics
import os
import math
from benchmark.evaluate_caption import eval_assigned_captions, plot_cap_eval
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
from benchmark.evaluate_semantic_instance import evaluate
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils.votenet_utils.eval_det import eval_det
from datasets.scannet200.scannet200_splits import (
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)
from datasets.utils import get_sem_inst_mappings
from torch_warmup_lr import WarmupLR

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools

# visualize on mesh
import open3d as o3d
from scannetpp.common.scene_release import ScannetppScene_Release
from scipy.spatial import KDTree

from third_party.cuda_utils.interpolate.cuda_utils import TrilinearInterpolateFeatures


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.num_classes = config.data.num_labels   

        self.filter_out_classes = config.data.train_dataset.filter_out_classes
        self.semid_to_instsemid, self.instsemid_to_semid = get_sem_inst_mappings(self.num_classes, self.filter_out_classes)

        self.semantic_class_names = None

        if config.data.validation_dataset.dataset_name == 'scannetpp':
            self.semantic_class_names = read_txt_list(config.data.semantic_classes_file)

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)

        if self.config.general.freeze_segmentor:
            for param in self.model.parameters():
                param.requires_grad = False
        ####### extra features for instseg #######
        if config.general.use_2d_feats_instseg and config.general.project_2d_feats_instseg:
            # linear and relu to project to a different dim
            self.feat2d_projector = nn.Sequential(
                nn.Linear(config.data.extra_feats_dim, config.general.project_2d_feats_instseg),
                nn.ReLU()
            )

        self.use_hidden_state_ndx = self.config.general.use_hidden_state_ndx

        ###### caption consistency ######
        if self.config.get('consistency', None):
            # make caption model output hidden states
            config.caption_model.output_hidden_states = True
            print(f'Output train hidden states in obj and part caption models')

            if self.config.consistency.classification:
                # classification heads for part and local caps
                print(f'>>>>>>> Use classification consistency for part-local captions')
                # caption model embdim -> num instance classes for obj and part captioners
                self.obj_cap_classifier = hydra.utils.instantiate(config.consistency.classification_model)
                self.part_cap_classifier = hydra.utils.instantiate(config.consistency.classification_model)
            if self.config.consistency.similarity:
                print(f'>>>>>>> Use similarity consistency for part-local captions')
                # projection models for part and local hidden states
                # create similar small GPT models without LM head
                self.obj_cap_projector = hydra.utils.instantiate(config.consistency.projection_model)
                self.part_cap_projector = hydra.utils.instantiate(config.consistency.projection_model)
        
        ####### caption models ########
        print(f'>>>>>>> Configure caption models >>>>>>>>')
        if config.general.gen_captions:
            self.caption_model = hydra.utils.instantiate(config.caption_model)
        if config.general.gen_part_captions:
            self.part_caption_model = hydra.utils.instantiate(config.caption_model)
            self.part_caption_model.tgt_prefix = 'part_'

        if config.general.obj_output_to_part:
            print(f'>>>>>>> Use obj caption hidden states as input to part captions >>>>>>>>')
            # obj model should output both hidden states
            self.caption_model.output_hidden_states = True
            self.caption_model.output_beam_hidden_states = True
            print(f'Output train and beam hidden states in obj caption model')
            self.part_caption_model.use_other_caption_feats = True
        elif config.general.part_output_to_obj:
            print(f'>>>>>>> Use part caption hidden states as input to obj captions >>>>>>>>')
            # part model should output both hidden states
            self.part_caption_model.output_hidden_states = True
            self.part_caption_model.output_beam_hidden_states = True
            print(f'Output train and beam hidden states in part caption model')
            self.caption_model.use_other_caption_feats = True
        if config.general.project_hidden_states:
            self.hidden_state_projector = nn.Sequential(
                nn.Linear(config.caption_model.embedding_size, config.caption_model.embedding_size), # keep the same dim
                nn.ReLU()
            )
        print(f'>>>>>>> Done configuring caption models >>>>>>>>')

        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label
        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        # create empty dicts to store preds and GT
        self.init_pred_gt_dicts()

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

        #### extra model features #### 
        self.feats_2d_model = None
        if config.general.proj_feat_2d:
            self.feats_2d_model = hydra.utils.instantiate(config.feats_2d_model)

        if self.config.general.project_queries_before_caption:
            layers = [nn.Linear(self.config.model.hidden_dim, self.config.general.project_queries_before_caption_dim),
                nn.LayerNorm(self.config.general.project_queries_before_caption_dim),
                nn.ReLU()]
            for _ in range(self.config.general.project_queries_before_caption_layers - 1):
                # add more layers if needed
                layers.extend([nn.Linear(self.config.general.project_queries_before_caption_dim, self.config.general.project_queries_before_caption_dim),
                    nn.LayerNorm(self.config.general.project_queries_before_caption_dim),
                    nn.ReLU()])
            self.query_projector = nn.Sequential(*layers)
        else:
            self.query_projector = nn.Identity()

        

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        return x
        
    def add_feats_2d(self, batch, extra_feats, output):
        '''
        extra feats: on all the points in the batch
        output[features3d]: add the features here
        '''
        coordinates = batch[0].coordinates # use this to get the indices of samples
        
        # join with backbone features
        new_feats = []

        for batch_ndx, backbone_feat in enumerate(output['features3d']):
            batch_mask = (coordinates[:, 0] == batch_ndx)
            extra_feat = extra_feats[batch_mask].to(self.device)

            if self.config.general.normalize_feats_2d: # normalize to same range as 3d feats
                bb_min, bb_max = backbone_feat.min(), backbone_feat.max()
                extra_feat = (extra_feat - extra_feat.min()) / (extra_feat.max() - extra_feat.min())
                extra_feat = extra_feat * (bb_max - bb_min) + bb_min

            new_feats.append(torch.cat([backbone_feat, extra_feat], dim=1))

        # now 96 + 2dfeatdim (dino=384 -> total 480)
        output['features3d'] = new_feats

        return output

    def prepare_feats_2d(self, batch, output, scene_ids):
        # proj 2d feats for all points
        batchdata = batch[0]
        extra_feats_3d = self.feats_2d_model(batchdata, scene_ids)
        # join with backbone features
        new_feats = []
        for backbone_feat, extra_feat in zip(output['features3d'], extra_feats_3d):
            if self.config.general.use_2d_feats_only:
                new_feats.append(extra_feat)
            else:
                if self.config.general.normalize_feats_2d: # normalize to same range as 3d feats
                    bb_min, bb_max = backbone_feat.min(), backbone_feat.max()
                    extra_feat = (extra_feat - extra_feat.min()) / (extra_feat.max() - extra_feat.min())
                    extra_feat = extra_feat * (bb_max - bb_min) + bb_min
                new_feats.append(torch.cat([backbone_feat, extra_feat], dim=1))
        # now 96 + 2dfeatdim (dino=384 -> total 480)
        output['features3d'] = new_feats

        return output

    def training_step(self, batch, batch_idx):
        # target and caption GT is already on GPU here
        data, target, file_names, cap_gt = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        # keep the extra feats separately if they were loaded in the dataset
        if self.config.data.extra_feats_dir: 
            extra_feats = data.features[:, 6:] 
            # color, rawcoords
            data.features = data.features[:, :6]

        raw_coordinates = None
        # order: color, normal, rawcoords, extrafeats
        # default is colors (0-2), rawcoords (3-5) = 3+3=6
        if self.config.data.add_raw_coordinates: # default true
            raw_coordinates = data.features[:, 3:6] # gets passed separately
            data.features = data.features[:, :3] # default: colors

        if self.config.general.use_2d_feats_instseg:
            if self.config.general.project_2d_feats_instseg:
                # NOTE: need to configure the final dim in data.in_channels
                extra_feats = self.feat2d_projector(extra_feats.to(self.device)).cpu()
            data.features = torch.cat([data.features, extra_feats], dim=1)

        # goes to device here, was on cpu till now!
        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        # use the training target, dont run instance seg model
        if self.config.general.eval_against_target:
            output = self.get_output_from_target(target)
        else:
            try:
                with torch.set_grad_enabled(not self.config.general.freeze_segmentor):
                    output = self.forward(
                        data,
                        point2segment=[
                            target[i]["point2segment"] for i in range(len(target))
                        ],
                        raw_coordinates=raw_coordinates,
                    )
                # training output: 'pred_logits', 'pred_masks', 'aux_outputs', 'sampled_coords', 'backbone_features', 'final_queries']
                # queries -> bsize, nqueries, 128
            except RuntimeError as run_err:
                print(run_err)
                if (
                    "only a single point gives nans in cross-attention"
                    == run_err.args[0]
                ):
                    return None
                else:
                    raise run_err

        # TODO: upsample the masks on segments (centers) to each voxel if specified
        # get instance segmentation loss
        try:
            # get the GT-pred assignments from hungarian matching
            losses, assignment = self.criterion(output, target, mask_type=self.mask_type)
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)


        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        total_loss = sum(losses.values())       
        ############ instance seg done #################3

        ############ captioning #################
        caption_loss, part_caption_loss = None, None

        # get centers of predicted queries
        output['query_centers'] = self.get_pred_query_centers(raw_coordinates, output, target) #1,nqueries,3 

        # prepare feats to be used by captioner
        # list of N, 96
        output['features3d'] = output['backbone_features'].decomposed_features

        if self.feats_2d_model:
            output = self.prepare_feats_2d(batch, output, file_names)
        elif self.config.general.use_2d_feats_caption: # concat existing feats
            output = self.add_feats_2d(batch, extra_feats, output)

        # project queries before part caption, could be a passthroguh func
        output['final_queries'] = self.query_projector(output['final_queries'])

        output_for_caption, assignment_for_caption = output, assignment
        output_for_caption['scene_ids'] = file_names           

        # run object caption first 
        if self.config.general.gen_captions and not self.config.general.part_output_to_obj:
            # forward pass caption model
            caption_output, caption_loss, caption_token_acc, caption_extra_output = self.caption_model(
                output_for_caption, assignment_for_caption, cap_gt, target, is_eval=False
            )
            if caption_loss is not None:
                logs['train_caption_loss'] = caption_loss.detach().cpu().item()
                logs['train_caption_token_acc'] = caption_token_acc.detach().cpu().item()
                total_loss += caption_loss

        if self.config.general.gen_part_captions:
            # use obj caption hidden state as input to part caps
            if self.config.general.obj_output_to_part:
                # has all the objid, sceneid, semid info required to match contexts with parts
                # objs for which there is no context will get a 0 feature
                output_for_caption['captioner_output'] = caption_extra_output
                
                if self.config.general.project_hidden_states and output_for_caption['captioner_output'] is not None and 'hidden_states' in output_for_caption['captioner_output']:
                    # cant assign to tuple!
                    output_for_caption['captioner_output']['hidden_states'] = list(output_for_caption['captioner_output']['hidden_states'])
                    # project only the last layer hidden state
                    output_for_caption['captioner_output']['hidden_states'][self.use_hidden_state_ndx] = self.hidden_state_projector(output_for_caption['captioner_output']['hidden_states'][self.use_hidden_state_ndx])

            part_caption_output, part_caption_loss, part_caption_token_acc, part_caption_extra_output = self.part_caption_model(
                output_for_caption, assignment_for_caption, cap_gt, target, is_eval=False
            )
            if part_caption_loss is not None:
                logs['train_part_caption_loss'] = part_caption_loss.detach().cpu().item()
                logs['train_part_caption_token_acc'] = part_caption_token_acc.detach().cpu().item()
                total_loss += part_caption_loss

        # run object caption second
        if self.config.general.part_output_to_obj:
            # has all the objid, sceneid, semid info required to match contexts with parts
            # objs for which there is no context will get a 0 feature
            output_for_caption['captioner_output'] = part_caption_extra_output
            #ERROR check that none of these are None
            if self.config.general.project_hidden_states and output_for_caption['captioner_output'] is not None and 'hidden_states' in output_for_caption['captioner_output']:
                output_for_caption['captioner_output']['hidden_states'] = list(output_for_caption['captioner_output']['hidden_states'])
                output_for_caption['captioner_output']['hidden_states'][self.use_hidden_state_ndx] = self.hidden_state_projector(output_for_caption['captioner_output']['hidden_states'][self.use_hidden_state_ndx])

            # forward pass caption model
            caption_output, caption_loss, caption_token_acc, caption_extra_output = self.caption_model(
                output_for_caption, assignment_for_caption, cap_gt, target, is_eval=False
            )
            if caption_loss is not None:
                logs['train_caption_loss'] = caption_loss.detach().cpu().item()
                logs['train_caption_token_acc'] = caption_token_acc.detach().cpu().item()
                total_loss += caption_loss
    
        if self.config.get('consistency', None) and caption_output and part_caption_output and caption_extra_output and part_caption_extra_output:
            # use part and local caption outputs to enforce consistency during training
            cons_loss, log_dict, _ = self.consistency_loss(caption_extra_output, part_caption_extra_output)
            logs['train_consistency_loss'] = cons_loss
            # log individual losses
            for key, value in log_dict.items():
                logs[f'train_{key}'] = value
            total_loss += cons_loss

        # all instance seg losses
        # segmentor frozen + caption_loss is None -> cant do backward
        if self.config.general.freeze_segmentor and caption_loss is None and part_caption_loss is None:
            return None
        
        # log at the end after storing everything!
        self.log_dict(logs)
        return total_loss

    def nce_loss_semantic(self, features1, features2, sem_labels):
        """
        Compute PointInfoNCE loss for two lists of feature vectors.

        Args:
            features1 (torch.Tensor): Tensor of shape (N, D) for first list of features.
            features2 (torch.Tensor): Tensor of shape (M, D) for second list of features.
            sem_labels (torch.Tensor): Tensor of shape (N,) for semantic labels of features1/features2

        Returns:
            torch.Tensor: Scalar loss value.
        """
        pos_sims = []
        neg_sims = []

        # no negatives to sample from
        if len(features1) == 1:
            return torch.tensor(0).to(self.device)

        # every object-part pair (has same semantics)
        positive_pairs = [(i, i) for i in range(min(features1.size(0), features2.size(0)))]
        # negative = i and j have different semantic labels
        negative_pairs = list(set([(i, j) for i in range(features1.size(0)) for j in range(features2.size(0)) if sem_labels[i] != sem_labels[j]]))

        max_negatives = self.config.consistency.similarity_cfg.max_negatives
        if max_negatives:
            # sample pairs 
            sample_ndx = random.sample(range(len(negative_pairs)), min(max_negatives, len(negative_pairs)))
            negative_pairs = [negative_pairs[i] for i in sample_ndx]

        if len(negative_pairs) == 0:
            # why does this happen - all objects are of the same class?
            return torch.tensor(0).to(self.device)

        # normalize features
        features1_norm = nn.functional.normalize(features1, p=2, dim=1)
        features2_norm = nn.functional.normalize(features2, p=2, dim=1)

        for (i, j) in positive_pairs:
            pos_sims.append(nn.functional.cosine_similarity(features1_norm[i].unsqueeze(0), features2_norm[j].unsqueeze(0)))

        for (i, j) in negative_pairs:
            neg_sims.append(nn.functional.cosine_similarity(features1_norm[i].unsqueeze(0), features2_norm[j].unsqueeze(0)))

        temp = self.config.consistency.similarity_cfg.contrastive_temp

        pos_sims = torch.cat(pos_sims, dim=0) / temp
        neg_sims = torch.cat(neg_sims, dim=0) / temp

        pos_exp = torch.exp(pos_sims)
        neg_exp = torch.exp(neg_sims)

        loss = -torch.log(pos_exp / (pos_exp + torch.sum(neg_exp)))
        return loss.mean()

    def hardest_margin_semantic(self, features1, features2, sem_labels):
        ''''
        Compute hardest-margin loss for features1 and features2 ->
            positives = entries in feat1 and feat2 with the same sem label
            negatives = feat1 and feat2 with different sem labels
            for a given feat1, use the feat2 that has the highest similarity (but from a different semantic class)
            as a negative and push its features away.
            loss = sum(dist(f1, f2) - margin_pos) + 0.5 max(margin_neg - min(dist(f1, f2)), 0)/num_neg + 0.5 (same for f2, f1 negatives)

        Args:
            features1 (torch.Tensor): Tensor of shape (N, D) for first list of features.
            features2 (torch.Tensor): Tensor of shape (M, D) for second list of features.
            sem_labels (torch.Tensor): Tensor of shape (N,) for semantic labels of features1/features2

        Returns:
            torch.Tensor: Scalar loss value.
        '''
        loss = torch.tensor(0).to(self.device)

        # no negatives to sample from
        if len(features1) == 1:
            return loss
        
        # margins
        m_pos = self.config.consistency.similarity_cfg.margin_pos
        m_neg = self.config.consistency.similarity_cfg.margin_neg
        dist_type = self.config.consistency.similarity_type

        # separate positive and negative losses
        pos_loss = 0
        neg_loss = 0
        num_pos, num_neg = 0, 0

        for f1_ndx in range(len(features1)):
            # positive is the same ndx in feat2
            pos_dist = self.feat_dist(features1[f1_ndx], features2[f1_ndx], dist_type)
            pos_loss += max(pos_dist - m_pos, 0)
            
            neg_indices = [f2_ndx for f2_ndx in range(len(features2)) if sem_labels[f1_ndx] != sem_labels[f2_ndx]]

            # no negatives for this sample -> all are from the same class, nothing to be done in this batch!
            if len(neg_indices) == 0:
                continue

            neg_feats = features2[neg_indices]
            # get the individual distances
            neg_dists = self.feat_dist(features1[f1_ndx], neg_feats, dist_type, reduction='none')
            if dist_type in ('l1', 'l2'):
                neg_dists = neg_dists.sum(dim=1)
            min_dist = neg_dists.min()
            neg_loss += max(m_neg - min_dist, 0)

            num_pos += 1
            num_neg += 1

        # average over all the positives and negatives
        eps = 1e-6
        loss = (pos_loss/(num_pos+eps)) + (neg_loss/(num_neg+eps))

        return loss
        
    def feat_dist(self, f1, f2, dist_type, reduction='mean'):
        '''
        f1, f2: single feature vectors
        '''
        if len(f1.shape) == 1:
            # make it 2 dims
            f1 = f1.unsqueeze(0)
        if len(f2.shape) == 1:
            f2 = f2.unsqueeze(0)

        # f1 could be smaller (len=1), repeat it to match f2
        if len(f1) == 1 and len(f2) > 1:
            f1 = f1.repeat(len(f2), 1)
        elif len(f2) == 1 and len(f1) > 1:
            f2 = f2.repeat(len(f1), 1)

        if dist_type == 'cosine':
            sim = nn.functional.cosine_similarity(f1, f2)
            if reduction == 'mean':
                sim = sim.mean()
            return sim
        elif dist_type == 'l1':
            return nn.functional.l1_loss(f1, f2, reduction=reduction)
        elif dist_type == 'l2':
            return nn.functional.mse_loss(f1, f2, reduction=reduction)

    def consistency_loss(self, caption_extra_output, part_caption_extra_output):
        loss = 0
        # store individual losses
        log_dict = {}
        outputs = {}
        
        # dont have have objs or parts in one of the batches
        if 'scene_ids' not in caption_extra_output or 'scene_ids' not in part_caption_extra_output:
            return loss, log_dict, outputs
        if len(caption_extra_output['scene_ids']) == 0 or len(part_caption_extra_output['scene_ids']) == 0:
            return loss, log_dict, outputs
        
        # both should have the same sceneids -> 
        obj_scene_ids = list(set(caption_extra_output['scene_ids']))

        obj_hidden_all = caption_extra_output['hidden_states'][self.use_hidden_state_ndx]
        part_hidden_all = part_caption_extra_output['hidden_states'][self.use_hidden_state_ndx]

        all_obj_ids = caption_extra_output['obj_ids'].to(self.device)
        all_part_obj_ids = part_caption_extra_output['obj_ids'].to(self.device)

        for scene_id in obj_scene_ids:
            # no parts in this scene
            if scene_id not in part_caption_extra_output['scene_ids']:
                continue

            # objs in this scene
            obj_in_scene = torch.BoolTensor([sid == scene_id for sid in caption_extra_output['scene_ids']]).to(self.device)
            obj_id_in_scene = all_obj_ids[obj_in_scene] # this shouldnt contain repetitions!
            
            # NOTE: assume that every part must belong to an object, every object doesnt have a part
            # but the assignment might be different for these objects? keep only the overlapping objects for which both obj and part captions are present
            part_in_scene = torch.BoolTensor([sid == scene_id for sid in part_caption_extra_output['scene_ids']]).to(self.device)
            part_obj_ids_in_scene = all_part_obj_ids[part_in_scene] # this shouldnt contain repetitions!

            # pick the objs for which part captions are present
            obj_id_has_part = torch.isin(obj_id_in_scene, part_obj_ids_in_scene)
            # partcaps for which there are objcaps
            part_id_has_obj = torch.isin(part_obj_ids_in_scene, obj_id_in_scene)

            # finally should get these many captions for comparison
            assert set(obj_id_in_scene[obj_id_has_part].tolist()) == set(part_obj_ids_in_scene[part_id_has_obj].tolist())

            # get the hidden states where objids match
            obj_hidden_selected = obj_hidden_all[obj_in_scene][obj_id_has_part] # nobj, seqlen+1, dim
            part_hidden_selected = part_hidden_all[part_in_scene][part_id_has_obj] # nobj, seqlen+1, dim -> both should have same shape

            if len(obj_hidden_selected) == 0 or len(part_hidden_selected) == 0:
                # no matching objects, ignore this scene
                continue

            assert obj_hidden_selected.shape == part_hidden_selected.shape

            if self.config.consistency.similarity:
                # put hidden states through projection models
                obj_cap_emb = self.obj_cap_projector(obj_hidden_selected) # n, embdim
                part_cap_emb = self.part_cap_projector(part_hidden_selected) # n, embdim

                # loss = similarity loss between embeddings = cosine similarityi
                sim_type = self.config.consistency.similarity_type

                if self.config.consistency.similarity_cfg.contrastive == 'nce_sem_neg':
                    # NCE contrastive loss with negative samples
                    # get the semantic labels corresponding to both the embeddings
                    sem_labels = caption_extra_output['sem_ids'][obj_in_scene][obj_id_has_part].to(self.device)
                    sim_loss = self.nce_loss_semantic(obj_cap_emb, part_cap_emb, sem_labels)
                elif self.config.consistency.similarity_cfg.contrastive == 'hardest_margin_sem_neg':
                    sem_labels = caption_extra_output['sem_ids'][obj_in_scene][obj_id_has_part].to(self.device)
                    sim_loss = self.hardest_margin_semantic(obj_cap_emb, part_cap_emb, sem_labels)
                else:
                    # simple loss with only positives - doesnt really work?
                    if sim_type == 'cosine':
                        sim_loss = nn.functional.cosine_similarity(obj_cap_emb, part_cap_emb, dim=1).mean()
                    elif sim_type == 'l1':
                        sim_loss = nn.functional.l1_loss(obj_cap_emb, part_cap_emb, reduction='mean')
                    elif sim_type == 'l2':
                        sim_loss = nn.functional.mse_loss(obj_cap_emb, part_cap_emb, reduction='mean')
                
                    # use pairs with different semantics as negative samples
                # add to total loss
                loss += self.config.consistency.loss_weights.similarity * sim_loss
                log_dict['consistency_similarity_loss'] = sim_loss

            if self.config.consistency.classification:
                # loss = classification loss between embeddings
                obj_cap_class = self.obj_cap_classifier(obj_hidden_selected)
                part_cap_class = self.part_cap_classifier(part_hidden_selected)
                # cross entropy loss both ways, detach each while using the logits of the other one
                obj_cap_loss = nn.functional.cross_entropy(obj_cap_class, part_cap_class.argmax(dim=1).detach())
                part_cap_loss = nn.functional.cross_entropy(part_cap_class, obj_cap_class.argmax(dim=1).detach())

                log_dict['consistency_obj_classification_loss'] = obj_cap_loss
                log_dict['consistency_part_classification_loss'] = part_cap_loss

                loss += self.config.consistency.loss_weights.classification * (obj_cap_loss + part_cap_loss)

                # additionally supervise with the gt semantic?
                if self.config.consistency.classification_cfg.use_gt_sem:
                    obj_sem_labels = caption_extra_output['sem_ids'][obj_in_scene][obj_id_has_part].to(self.device)
                    part_sem_labels = part_caption_extra_output['sem_ids'][part_in_scene][part_id_has_obj].to(self.device)

                    obj_sem_loss = nn.functional.cross_entropy(obj_cap_class, obj_sem_labels)
                    part_sem_loss = nn.functional.cross_entropy(part_cap_class, part_sem_labels)

                    log_dict['consistency_obj_semantic_loss'] = obj_sem_loss
                    log_dict['consistency_part_semantic_loss'] = part_sem_loss

                    # log the semantic accuracy consistency_obj_aux_sem_acc
                    obj_sem_acc = (obj_cap_class.argmax(dim=1) == obj_sem_labels).float().mean()
                    part_sem_acc = (part_cap_class.argmax(dim=1) == part_sem_labels).float().mean()

                    log_dict['consistency_obj_aux_sem_acc'] = obj_sem_acc
                    log_dict['consistency_part_aux_sem_acc'] = part_sem_acc

                    loss += self.config.consistency.loss_weights.classification_gt * (obj_sem_loss + part_sem_loss)

            # TODO: store all the semantic preds on the objects to be saved out later

        return loss, log_dict, outputs
    

    def get_confident_output_for_caption(self, output, assignment):
        output_for_caption = {}
        # keep these many top predictions
        n_keep = self.config.general.caption_confident_instances

        pred_scores = torch.functional.F.softmax(output['pred_logits'], dim=-1)[..., :-1]

        bsize = output['pred_logits'].shape[0]

        # exclude background class
        num_classes = self.model.num_classes - 1

        final_queries_for_caption = []
        assignment_for_caption = []

        for bid in range(bsize):
            # 100, nclasses
            sample_scores = pred_scores[bid]
            # ascending order? doesnt matter
            # flatten = topk scores across classes, across queries
            _, indices = sample_scores.flatten().topk(n_keep, sorted=True)
            # get indices into original array
            # keep everything according to indices
            keep_query_indices = indices // num_classes

            # caption model just needs final queries
            # nkeep, 128
            keep_final_queries = output['final_queries'][bid][keep_query_indices]
            final_queries_for_caption.append(keep_final_queries)

            # remap assignment indices to new queries
            sample_query_ndx, sample_gt_ndx = assignment[bid]
            sample_new_query_ndx, sample_new_gt_ndx = [], []

            for q_ndx, gt_ndx in zip(sample_query_ndx, sample_gt_ndx):
                if q_ndx in keep_query_indices:
                    # new query index indices into new queries
                    new_q_ndx = keep_query_indices.tolist().index(q_ndx)
                    sample_new_query_ndx.append(new_q_ndx)
                    sample_new_gt_ndx.append(gt_ndx)

            sample_new_query_ndx = torch.LongTensor(sample_new_query_ndx).to(self.device)
            sample_new_gt_ndx = torch.LongTensor(sample_new_gt_ndx).to(self.device)

            assignment_for_caption.append((sample_new_query_ndx, sample_new_gt_ndx))

        # B, nkeep, 128
        output_for_caption['final_queries'] = torch.stack(final_queries_for_caption, dim=0)

        return output_for_caption, assignment_for_caption


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id,
               assignment=None, obj_ids_orig=None, ious=None):
        # save to the same dir as checkpoints, etc
        root_path = self.config.general.save_dir
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        print(f'*'*50)
        print(f'Saving instance mask preds to: {pred_mask_path}')
        print(f'*'*50)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

        if assignment is not None and obj_ids_orig is not None:
            # get the obj ids according to the GT ndx
            all_gt_ndx = assignment[1]
            all_obj_ids = []
            for gt_ndx in all_gt_ndx:
                # -1 -> no query assignment
                if gt_ndx == -1:
                    all_obj_ids.append(-1)
                else:
                    all_obj_ids.append(obj_ids_orig[gt_ndx.item()].item())
            
            # save to file
            out_path = Path(base_path) / 'assigned_gt_obj_ids' / f'{file_name}.json'
            # makedir
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # write to json file
            with open(out_path, 'w') as f:
                json.dump(all_obj_ids, f)
        if ious is not None:
            out_path = Path(base_path) / 'ious' / f'{file_name}.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(ious, f)

    def training_epoch_end(self, outputs):
        if not outputs:
            return
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        # val_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        # results = {"val_loss_mean": val_loss}
        self.test_epoch_end(outputs)

    def save_visualizations(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        sorted_heatmaps=None,
        query_pos=None,
        backbone_features=None,
    ):
        # store the mean of the full res coords to vis on mesh
        full_res_coords_mean = full_res_coords.mean(axis=0)

        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []

        if "labels" in target_full:
            # get instance colors
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask) in enumerate(
                zip(target_full["labels"], target_full["masks"])
            ):
                if label == 255:
                    continue

                mask_tmp = mask.detach().cpu().numpy()
                # full res coords of points in the instance
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue
                
                gt_pcd_pos.append(mask_coords)
                # get the min and max coords of the instance
                mask_coords_min = full_res_coords[
                    mask_tmp.astype(bool), :
                ].min(axis=0)
                mask_coords_max = full_res_coords[
                    mask_tmp.astype(bool), :
                ].max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append(
                    {
                        "position": mask_coords_middle,
                        "size": size,
                        "color": self.validation_dataset.map2color([label])[0],
                    }
                )

                gt_pcd_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        gt_pcd_pos[-1].shape[0], 1
                    )
                )
                gt_inst_pcd_color.append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(gt_pcd_pos[-1].shape[0], 1)
                )

                gt_pcd_normals.append(
                    original_normals[mask_tmp.astype(bool), :]
                )

            # GT instance points and colors
            gt_pcd_pos = np.concatenate(gt_pcd_pos)
            gt_pcd_normals = np.concatenate(gt_pcd_normals)
            gt_pcd_color = np.concatenate(gt_pcd_color)
            gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        if self.config.general.save_viz_pyviz:
            v = vis.Visualizer()

            v.add_points(
                "RGB Input",
                full_res_coords,
                colors=original_colors,
                normals=original_normals,
                visible=True,
                point_size=point_size,
            )

            if backbone_features is not None:
                v.add_points(
                    "PCA",
                    full_res_coords,
                    colors=backbone_features,
                    normals=original_normals,
                    visible=False,
                    point_size=point_size,
                )

            if "labels" in target_full:
                v.add_points(
                    "Semantics (GT)",
                    gt_pcd_pos,
                    colors=gt_pcd_color,
                    normals=gt_pcd_normals,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size,
                )
                v.add_points(
                    "Instances (GT)",
                    gt_pcd_pos,
                    colors=gt_inst_pcd_color,
                    normals=gt_pcd_normals,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size,
                )

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []
        pred_heatmap_values = []

        # d = 0 only, sorted_masks is a list of 1 element
        # with 100 masks on full res coords nfullres, 100
        for did in range(len(sorted_masks)):
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        max(1, sorted_masks[did].shape[1])
                    )
                )
            )

            for i in reversed(range(sorted_masks[did].shape[1])):
                mask_coords = full_res_coords[
                    sorted_masks[did][:, i].astype(bool), :
                ]
                if len(mask_coords) == 0:
                    continue

                mask_normals = original_normals[
                    sorted_masks[did][:, i].astype(bool), :
                ]

                label = sort_classes[did][i]

                # viz only the points within masks! 
                pred_coords.append(mask_coords)
                pred_normals.append(mask_normals)

                # sem colors for each of these points
                pred_sem_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        mask_coords.shape[0], 1
                    )
                )
                # instance colors for each of these points
                pred_inst_color.append(
                    instances_colors[i % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(mask_coords.shape[0], 1)
                )

                # accumulate heatmap values for the predicted instances
                pred_heatmap_values.append(sorted_heatmaps[did][:, i][
                    sorted_masks[did][:, i].astype(bool)
                ])

            # have something to show
            if len(pred_coords) > 0:
                pred_coords = np.concatenate(pred_coords)
                pred_normals = np.concatenate(pred_normals)
                pred_sem_color = np.concatenate(pred_sem_color)
                pred_inst_color = np.concatenate(pred_inst_color)
                pred_heatmap_values = np.concatenate(pred_heatmap_values)
                
                if self.config.general.save_viz_pyviz:
                    v.add_points(
                        "Semantics (Mask3D)",
                        pred_coords,
                        colors=pred_sem_color,
                        normals=pred_normals,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )
                    v.add_points(
                        "Instances (Mask3D)",
                        pred_coords,
                        colors=pred_inst_color,
                        normals=pred_normals,
                        visible=False,
                        alpha=0.8,
                        point_size=point_size,
                    )

        if self.config.general.save_viz_pyviz:
            v.save(
                f"{self.config['general']['save_dir']}/visualizations/{file_name}"
            )    
        if self.config.general.save_viz_on_original_meshes:
            scene_id = file_name
            scene = ScannetppScene_Release(scene_id, self.config.general.spp_data_dir)
            mesh_path = scene.scan_mesh_path
            # read original mesh 
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            mesh_vertices = np.asarray(mesh.vertices)
            mesh_vertices_tree = KDTree(mesh_vertices)

            # init to gray (0.5)
            pred_inst_vtx_color = np.ones_like(mesh_vertices) * 0.5
            # pred coords, pred_inst_color
            # add back the mean to get coords in the original space
            _, ndx = mesh_vertices_tree.query(pred_coords + full_res_coords_mean)
            # ndx = for each pred coord, which is the nearest mesh vertex
            # get colors onto mesh vertices
            pred_inst_vtx_color[ndx] = pred_inst_color / 255.0

            mesh_out_dir = Path(self.config.general.save_dir) / "visualizations" / file_name
            mesh_out_dir.mkdir(parents=True, exist_ok=True)

            # copy the original mesh
            pred_viz_mesh = deepcopy(mesh)
            pred_viz_mesh.vertex_colors = o3d.utility.Vector3dVector(pred_inst_vtx_color)
            out_path = mesh_out_dir / f"pred_inst.ply"
            o3d.io.write_triangle_mesh(str(out_path), pred_viz_mesh)
            print(f"Saved pred inst mesh to {out_path}")
            # translate with mean to normalize space and save with _norm suffix
            pred_viz_mesh.translate(-full_res_coords_mean)
            out_path = mesh_out_dir / f"pred_inst_norm.ply"
            o3d.io.write_triangle_mesh(str(out_path), pred_viz_mesh)
            print(f"Saved pred inst mesh to {out_path}")
            
            # viz heatmap of each mask separately on the whole PC (not just within the mask where logits > 0.5)
            # save to heatmaps dir
            heatmaps_dir = mesh_out_dir / 'heatmaps'
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            n_masks = sorted_heatmaps[0].shape[1]

            # viz each heatmap
            # need another ndx query from fullrescoords to mesh vertices
            _, coords_ndx = mesh_vertices_tree.query(full_res_coords + full_res_coords_mean)

            for mask_ndx in tqdm(range(n_masks), desc="Saving heatmap meshes"):
                heatmap_vals = sorted_heatmaps[0][:, mask_ndx]
                heatmap_viz_color = np.ones_like(mesh_vertices) * 0.5
                heatmap_viz_color[coords_ndx] = 0
                heatmap_viz_color[coords_ndx, 0] = heatmap_vals
                heatmap_viz_mesh = deepcopy(mesh)
                heatmap_viz_mesh.vertex_colors = o3d.utility.Vector3dVector(heatmap_viz_color)
                out_path = heatmaps_dir / f"heatmap_{mask_ndx}.ply"
                o3d.io.write_triangle_mesh(str(out_path), heatmap_viz_mesh)
                print(f"Saved heatmap mesh to {out_path}")

            # viz heatmaps 0-1 in the red channel
            heatmap_viz_color = np.ones_like(mesh_vertices) * 0.5
            # set all channels to 0 
            heatmap_viz_color[ndx] = 0
            # set red channel to heatmap values
            heatmap_viz_color[ndx, 0] = pred_heatmap_values # already a np array
            heatmap_viz_mesh = deepcopy(mesh)
            heatmap_viz_mesh.vertex_colors = o3d.utility.Vector3dVector(heatmap_viz_color)
            out_path = mesh_out_dir / f"pred_heatmap.ply"
            o3d.io.write_triangle_mesh(str(out_path), heatmap_viz_mesh)
            print(f"Saved pred heatmap mesh to {out_path}")
            # translate with mean to normalize space and save with _norm suffix
            heatmap_viz_mesh.translate(-full_res_coords_mean)
            out_path = mesh_out_dir / f"pred_heatmap_norm.ply"
            o3d.io.write_triangle_mesh(str(out_path), heatmap_viz_mesh)
            print(f"Saved pred heatmap mesh to {out_path}")
            
            # repeat for GT colors
            gt_inst_vtx_color = np.ones_like(mesh_vertices) * 0.5
            _, ndx = mesh_vertices_tree.query(gt_pcd_pos + full_res_coords_mean)
            gt_inst_vtx_color[ndx] = gt_pcd_color / 255.0
            gt_viz_mesh = deepcopy(mesh)
            gt_viz_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_inst_vtx_color)
            out_path = mesh_out_dir / f"gt_inst.ply"    
            o3d.io.write_triangle_mesh(str(out_path), gt_viz_mesh)
            print(f"Saved gt inst mesh to {out_path}")
            # translate with mean to normalize space and save with _norm suffix
            gt_viz_mesh.translate(-full_res_coords_mean)
            out_path = mesh_out_dir / f"gt_inst_norm.ply"
            o3d.io.write_triangle_mesh(str(out_path), gt_viz_mesh)
            print(f"Saved gt inst mesh to {out_path}")


    def get_output_from_target(self, target):
        # copy target into output format
        # output: 'pred_logits', 'pred_masks', 'aux_outputs', 'sampled_coords', 'backbone_features'
        # pred logits: bsize, num queries, num classes + 1
        # pred_masks: list [(nsegments, nqueries) for each scene] -> for each segment, which object does it belong to?
        # aux_outputs: list of dicts, each dict has 'pred_logits' and 'pred_masks', same shape as above, just copy it 
        # sampled_coords: bsize, nqueries, 3 = ?? not used anywhere?
        # backbone features: only for viz, not requried
        output = {
            'pred_logits': [], #doesnt matter if its a list?
            'pred_masks': [],
            'aux_outputs': [],
            'sampled_coords': None,
            'backbone_features': None
        }
        for sample_ndx in range(len(target)):
            # create fake logits with 1 extra class (gets removed later)
            sample_target = target[sample_ndx]
            # nobjects, nclasses
            fake_logits = torch.nn.functional.one_hot(sample_target['labels'], self.num_classes + 1).float()
            output['pred_logits'].append(fake_logits)
            # nsegment, nobject
            fake_pred_mask = sample_target['segment_mask'].T.float()
            output['pred_masks'].append(fake_pred_mask)
            aux_output = {
                'pred_logits': fake_logits,
                'pred_masks': fake_pred_mask
            }
            output['aux_outputs'].append(aux_output)

        # concat logits, this should add a new bsize dimension
        # TODO: cant do this with bsize > 1
        output['pred_logits'] = torch.stack(output['pred_logits'], dim=0)

        return output

    def get_pred_query_centers(self, raw_coordinates, output, target):
        bsize = output['pred_logits'].shape[0]
        query_centers = []

        start_npts = 0
        # get predicted mask centers for each query
        for batch_ndx in range(bsize):
            sample_query_centers = []
            # nvoxels, nqueries
            sample_npts = len(target[batch_ndx]['point2segment'])
            lowres_mask = output['pred_masks'][batch_ndx].detach().cpu()[target[batch_ndx]['point2segment'].cpu()].sigmoid() > 0.5
            # raw coords has all the points stacked together!
            sample_pts = raw_coordinates[start_npts:start_npts+sample_npts]
            start_npts += sample_npts

            for query_mask in lowres_mask.T:
                if query_mask.sum() == 0:
                    # no points assigned to this query
                    center = torch.Tensor([0,0,0])
                else:
                    pts = sample_pts[query_mask.bool()] 
                    center = pts.mean(axis=0)

                sample_query_centers.append(torch.Tensor(center))

            # nquery, 3
            query_centers.append(torch.stack(sample_query_centers, dim=0))
        
        # bsize, nqueries, 3
        query_centers = torch.stack(query_centers, dim=0)

        return query_centers
        

    def eval_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader=0 if only datase being used
        data, target, file_names, cap_gt = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        # eg. 131k, 3 = original point coordinates, not normalized!
        original_coordinates = data.original_coordinates

        # eg. 107k, 3 = coordinates of voxels
        if len(data.coordinates) == 0:
            return 0.0

        # 107k, 3 -> normalized coordinates of points corresponding to voxels
        raw_coordinates = None

        # keep the extra feats separately if they were loaded in the dataset
        if self.config.data.extra_feats_dir: 
            extra_feats = data.features[:, 6:] 
            # color, rawcoords
            data.features = data.features[:, :6]

        # order: color, normal, rawcoords, extrafeats
        # default is colors (0-2), rawcoords (3-5) = 3+3=6
        if self.config.data.add_raw_coordinates: # default true
            raw_coordinates = data.features[:, 3:6] # gets passed separately
            data.features = data.features[:, :3] # default: colors

        if self.config.general.use_2d_feats_instseg:
            if self.config.general.project_2d_feats_instseg:
                extra_feats = self.feat2d_projector(extra_feats.to(self.device)).cpu()
            data.features = torch.cat([data.features, extra_feats], dim=1)

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )
        losses = {}
        # use the training target, dont run instance seg model
        if self.config.general.eval_against_target:
            output = self.get_output_from_target(target)
        else:
            # do instance seg forward pass
            try:
                output = self.forward(
                    data,
                    point2segment=[
                        target[i]["point2segment"] for i in range(len(target))
                    ],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
            except RuntimeError as run_err:
                print(run_err)
                if (
                    "only a single point gives nans in cross-attention"
                    == run_err.args[0]
                ):
                    return None
                else:
                    raise run_err
        
        output['query_centers'] = self.get_pred_query_centers(raw_coordinates, output, target) #bsize,nqueries,3 

        if self.config.general.viz_attn_mask: # self attention masks
            # assume bsize 1
            out_dir = Path(self.config.general.save_dir) / "attn_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_data = {
                'attn_mask': output['attn_masks'][0].detach().cpu().numpy(), # numvox, nqueries attn (inverted!)
                'scene_id': file_names[0],
                'raw_coords': raw_coordinates.detach().cpu().numpy(), # voxel coords? doesnt have a batch dim for bsize1
                'orig_coords': original_coordinates[0], # original coords?
                'sampled_coords': output['sampled_coords'][0], # initial location of each query
                'query_centers': output['query_centers'],
            }
            out_path = out_dir / f"{file_names[0]}_selfattn_query_3d.pth"
            torch.save(save_data, out_path)

        # compute losses and get assignments
        # need assignment for captioning!
        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)
            try:
                # val loss and assignment
                losses, assignment = self.criterion(
                    output, target, mask_type=self.mask_type
                )
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)
        
        rescaled_pca, backbone_features = None, None
        if self.config.general.save_visualizations or self.config.general.viz_backbone_feats:
            # get features for viz
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            # 3 channels
            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            # rescale to RGB range -> can be used as colors of points
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )

        ########### instance segmentation done ############

        ########### captioning #############
        # prepare feats to be used by captioner
        # list of N, 96
        output['features3d'] = output['backbone_features'].decomposed_features

        if self.feats_2d_model:
            output = self.prepare_feats_2d(batch, output, file_names)
        elif self.config.general.use_2d_feats_caption: # concat existing feats
            output = self.add_feats_2d(batch, extra_feats, output)

        # project queries before caption, could be a passthroguh func
        # TODO: this destroys the original queries, do we still need them? then make a different copy
        output['final_queries'] = self.query_projector(output['final_queries'])
        output['scene_ids'] = file_names

        # run caption model if requried
        caption_output, part_caption_output = None, None

        # run obj cap first
        if self.config.general.gen_captions and not self.config.general.part_output_to_obj:
            # forward pass caption model
            caption_output, caption_loss, caption_token_acc, caption_extra_output = self.caption_model(
                output, assignment, cap_gt, target, is_eval=True
            )
            if caption_loss is not None:
                # "val_" is added and detach etc done below
                losses['caption_loss'] = caption_loss
                # add token acc to "losses", doesnt matter?
                losses['caption_token_acc'] = caption_token_acc

            if self.config.general.viz_attn_mask or self.config.general.viz_backbone_feats:
                out_dir = Path(self.config.general.save_dir) / "attn_masks"
                out_dir.mkdir(parents=True, exist_ok=True)
                save_data = caption_extra_output
                # logits, not bool
                lowres_mask_logits = output['pred_masks'][0].detach().cpu()[target[0]['point2segment'].cpu()]
                lowres_mask = lowres_mask_logits.sigmoid() > 0.5
                # add hires masks for each query to viz
                hires_mask = self.get_full_res_mask(lowres_mask_logits, inverse_maps[0], target_full[0]['point2segment'])

                caption_extra_output.update({
                    'scene_id': file_names[0],
                    'inverse_map': inverse_maps[0].detach().cpu().numpy(),
                    'point2segment_full': target_full[0]['point2segment'].detach().cpu().numpy(),
                    'lowres_mask': lowres_mask.detach().cpu().numpy(),
                    'lowres_coords': raw_coordinates.detach().cpu().numpy(),
                    'hires_mask': hires_mask.detach().cpu().numpy(),
                    'hires_coords': original_coordinates[0],
                    'eval_pred_caps': caption_output,
                    'backbone_features': backbone_features,
                    'rescaled_pca': rescaled_pca
                })

                out_path = out_dir / f"{file_names[0]}_crossattn_caption_3d.pth"
                torch.save(save_data, out_path)
                print(f'Saved attn data to {out_path}, exiting')
                import sys; sys.exit(0)

        if self.config.general.gen_part_captions:
            # use obj caption hidden state as input to part caps
            if self.config.general.obj_output_to_part:
                # has all the objid, sceneid, semid info required to match contexts with parts
                # objs for which there is no context will get a 0 feature
                output['captioner_output'] = caption_extra_output
                if self.config.general.project_hidden_states and output['captioner_output'] is not None and 'val_hidden_states' in output['captioner_output']:
                    # NOTE: during validation, use val hidden states from beam search
                    for bid in range(len(output['captioner_output']['val_hidden_states'])):
                        output['captioner_output']['val_hidden_states'][bid] = list(output['captioner_output']['val_hidden_states'][bid])
                        output['captioner_output']['val_hidden_states'][bid][self.use_hidden_state_ndx] = self.hidden_state_projector(output['captioner_output']['val_hidden_states'][bid][self.use_hidden_state_ndx])

            part_caption_output, part_caption_loss, part_caption_token_acc, part_caption_extra_output = self.part_caption_model(
                output, assignment, cap_gt, target, is_eval=True
            )
            if part_caption_loss is not None:
                losses['part_caption_loss'] = part_caption_loss
                losses['part_caption_token_acc'] = part_caption_token_acc

        # run obj caption second # TODO: better way to do this instead of having it twice?
        if self.config.general.part_output_to_obj:
            # has all the objid, sceneid, semid info required to match contexts with parts
            # objs for which there is no context will get a 0 feature
            output['captioner_output'] = part_caption_extra_output
            if self.config.general.project_hidden_states and output['captioner_output'] is not None and 'val_hidden_states' in output['captioner_output']:
                for bid in range(len(output['captioner_output']['val_hidden_states'])):
                    output['captioner_output']['val_hidden_states'][bid] = list(output['captioner_output']['val_hidden_states'][bid])
                    output['captioner_output']['val_hidden_states'][bid][self.use_hidden_state_ndx] = self.hidden_state_projector(output['captioner_output']['val_hidden_states'][bid][self.use_hidden_state_ndx])

            # forward pass caption model
            caption_output, caption_loss, caption_token_acc, caption_extra_output = self.caption_model(
                output, assignment, cap_gt, target, is_eval=True
            )
            if caption_loss is not None:
                # "val_" is added and detach etc done below
                losses['caption_loss'] = caption_loss
                # add token acc to "losses", doesnt matter?
                losses['caption_token_acc'] = caption_token_acc

        if self.config.get('consistency', None) and caption_output and part_caption_output and caption_extra_output and part_caption_extra_output:
            # use part and local caption outputs to enforce consistency during training
            cons_loss, log_dict, _ = self.consistency_loss(caption_extra_output, part_caption_extra_output)
            losses['consistency_loss'] = cons_loss # gets detached, etc later
            # log individual losses
            for key, value in log_dict.items():
                losses[key] = value

        # calculate more stuff required for box AP and AP and store them
        # save the predictions to file 
        self.eval_instance_step(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            backbone_features=rescaled_pca
                if self.config.general.save_visualizations
                else None,
            caption_output=caption_output,
            cap_gt=cap_gt,
            assignment=assignment,
            part_caption_output=part_caption_output
        )
        
        if self.config.data.test_mode != "test":
            ret_dict = {}
            for k, v in losses.items():
                if torch.is_tensor(v):
                    ret_dict[f"val_{k}"] = v.detach().cpu().item()
                else:
                    ret_dict[f"val_{k}"] = v

            return ret_dict
        else:
            return 0.0

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, dataloader_idx)

    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        # mask: nvoxel, nqueries
        # inverse_map: npoints, ->nvoxel mapping
        # point2segment_full: npoints, -> nsegments mapping
        mask = mask.detach().cpu()[inverse_map]  # full res, npoint,nquery

        if self.eval_on_segments and is_heatmap == False: #True default
            mask = scatter_mean( # get average logit within each segment
                mask, point2segment_full, dim=0
            )  # go to segments, nqueries
            mask = (mask > 0.5).float() # binarize 
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # go back to full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None,
        assignment=None
    ):
        '''
        mask_cls: logits for each mask
        mask_pred: binary logits for the mask on voxels
        num_queries: number of queries
        num_classes: number of instance classes
        device: 
        assignment: [queryndx], [gt_ndx]

        return:
            score: score for each of topk masks (order is changed wrt the input)
            result_pred_mask: binary mask for each of topk masks (order is changed wrt the input)
            classes: class for each mask
            heatmap: heatmap for each mask
            assignment_sorted: sorted assignment

        '''
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.config.general.topk_per_image != -1:
            if self.config.general.eval_against_target:
                actual_num_vals = len(mask_cls.flatten(0, 1))
                topk_per_image = min(actual_num_vals, self.config.general.topk_per_image)
            else:
                # use the specified value
                # there are always 100 predictions made
                topk_per_image = self.config.general.topk_per_image
            # topk per image is 100 by default
            # when evaluating against GT, keep everything 
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        # here, heatmap = mask preds in 0-1 range?
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query
        
        assignment_topk = None
        if assignment:
            query_ndx, gt_ndx = assignment
            new_query_ndx, new_gt_ndx = [], []
            # assignment doesnt have all the queries (<100) -> not all are assigned to GT
            # while input queries are all -> pick accordingly!
            for topk_ndx in topk_indices:
                if topk_ndx in query_ndx: # both are tensors
                    new_query_ndx.append(topk_ndx.item())
                    pos_in_query_ndx = torch.where(query_ndx==topk_ndx)[0].item()
                    new_gt_ndx.append(gt_ndx[pos_in_query_ndx].item())
                else:
                    # add dummy -1 entries to match the same number of elements as the masks
                    new_query_ndx.append(-1)
                    new_gt_ndx.append(-1)
            assignment_topk = [torch.LongTensor(new_query_ndx).to(device), torch.LongTensor(new_gt_ndx).to(device)]

        return score, result_pred_mask, classes, heatmap, assignment_topk

    def interpolate_mask_logits_custom(self, seg_mask_logits, point2segment, coords):
        '''
        Use custom CUDA kernel to interpolate mask logits
        mask on chunk segments -> get location of each segment + logit on each segment
        trilerp logits to get logits on points using MinkowskiInterpolation

        seg_mask_logits: numsegs x 100 = logit for masks on segments
        point2segment: npoints = segid for each point
        coords: npoints x 3 = coordinates of points
        '''
        # move inputs to GPU
        seg_mask_logits = seg_mask_logits.to(self.device)

        # move coords to gpu, do everything on gpu. rest is already on gpu
        coords = coords.to(self.device)

        # get the coordinate of centroid of each segment = scatter_mean all coords in each segment
        seg_centroids = scatter_mean(coords, point2segment, dim=0) # nsegments, 3
        # create minkowski sparsetensor with locations = segcentroids, values = seg_mask_logits

        input_stensor = ME.SparseTensor(
            features=seg_mask_logits,
            # voxel size = the chunk size -> have features on for each chunk
            # returns integer coordinates!
            coordinates=ME.utils.batched_coordinates([seg_centroids // self.config.general.seg_chunk_size]).to(self.device), 
            # aggregate logits again on voxels
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE 
        )

        interp = TrilinearInterpolateFeatures()
        # get features on each voxel, voxel is smaller than chunk
        # query_features, interpolation_indices, interpolation_weights, accum_voxel_weights
        # scale query coords to the same scale as input, dont do any scaling later
        query_coords = coords / self.config.general.seg_chunk_size
        # add batch dim to query coords (all 0s, only 1 sample)
        query_coords = torch.cat([torch.zeros(len(query_coords), 1).to(self.device), query_coords], dim=1)
        
        interp_logits, _, _, _ = interp(input_stensor, query_coords)
        return interp_logits

    def interpolate_mask_logits(self, seg_mask_logits, point2segment, coords):
        '''
        mask on chunk segments -> get location of each segment + logit on each segment
        trilerp logits to get logits on points using MinkowskiInterpolation

        seg_mask_logits: numsegs x 100 = logit for masks on segments
        point2segment: npoints = segid for each point
        coords: npoints x 3 = coordinates of points
        '''
        # move coords to gpu, do everything on gpu. rest is already on gpu
        coords = coords.to(self.device)

        # get the coordinate of centroid of each segment = scatter_mean all coords in each segment
        seg_centroids = scatter_mean(coords, point2segment, dim=0) # nsegments, 3
        # create minkowski sparsetensor with locations = segcentroids, values = seg_mask_logits
        
        input_stensor = ME.SparseTensor(
            features=seg_mask_logits,
            # scale to the chunk size so that there is 1 logit value per chunk
            # returns integer coordinates!
            coordinates=ME.utils.batched_coordinates([seg_centroids / self.config.general.seg_chunk_size]).to(self.device), 
            # aggregate logits again on voxels?
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE 
        )
        # then interpolate to get logits at "coords"
        interp = ME.MinkowskiInterpolation().to(self.device)

        # original float output coordinates -> how is the scaling known between float coards and voxel coords?
        # output float coards should be in the same scale as the input voxel coords
        # subtract half the chunk size because the input coords have been quantized
        output_coords = (coords / self.config.general.seg_chunk_size)  #- 0.5 * self.config.general.seg_chunk_size

        # viz the quantized input coords in red and the output coords in blue
        pc = o3d.geometry.PointCloud()
        points = np.concatenate([input_stensor.decomposed_coordinates[0].cpu().numpy(), output_coords.cpu().numpy()], axis=0)
        colors = np.concatenate([np.ones_like(input_stensor.decomposed_coordinates[0].cpu().numpy()) * [1, 0, 0], np.ones_like(output_coords.cpu().numpy()) * [0, 0, 1]], axis=0)
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        out_path = Path('/rhome/cyeshwanth/Mask3D') / "voxel_inputs_outputs.ply"
        o3d.io.write_point_cloud(str(out_path), pc)
        print(f"Saved voxel inputs and outputs to {out_path}")

        # add batch dim to output coords (all 0s, only 1 sample)
        output_coords = torch.cat([torch.zeros(len(output_coords), 1).to(self.device), output_coords], dim=1)
        interp_logits = interp(input_stensor, output_coords)

        # viz segment centroids in red and output coordscoords in blue in the same point cloud
        pc = o3d.geometry.PointCloud()
        points = np.concatenate([seg_centroids.cpu().numpy(), coords.cpu().numpy()], axis=0)
        colors = np.concatenate([np.ones_like(seg_centroids.cpu().numpy()) * [1, 0, 0], np.ones_like(coords.cpu().numpy()) * [0, 0, 1]], axis=0)
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        out_path = Path('/rhome/cyeshwanth/Mask3D') / "seg_centroids_coords.ply"
        o3d.io.write_point_cloud(str(out_path), pc)
        print(f"Saved segment centroids and output coords to {out_path}")
        return interp_logits

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
        cap_gt=None,
        assignment=None,
        caption_output=None,
        part_caption_output=None
    ):
        label_offset = self.validation_dataset.label_offset
        # get the predictions from model outputs -> first add aux outputs
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )
        # predictions from last layer / final predictions
        # ignore the last element of logit
        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_unsorted_pred_masks = []

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()
        all_assignment_topk = []

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            # each sample in the batch = one scenes
            # first_full_res: False -> do this
            if not first_full_res:
                # default: true, do this 
                # get masks onto voxels
                if self.model.train_on_segments:
                    if self.config.general.chunk_seg_upsample_trilerp:
                        if self.config.general.chunk_seg_interp_type == "custom":
                            interp_func = self.interpolate_mask_logits_custom
                        elif self.config.general.chunk_seg_interp_type == "mink":
                            interp_func = self.interpolate_mask_logits
                        else:
                            raise ValueError(f"Invalid interpolation type: {self.config.general.chunk_seg_interp_type}")
                        
                        masks = interp_func(
                            # keep on gpu and do the interp
                            prediction[self.decoder_id]["pred_masks"][bid].detach(), 
                            target_low_res[bid]["point2segment"],
                            # corresponding coordinates of voxels
                            raw_coords # this is on cpu!
                        ).cpu() # finally bring back to cpu
                    else:
                        # use the same logit for the whole segment
                        # masks on the voxels -> logits, not actual masks
                        # voxels, 100 logits
                        # eg: 107k x 100 corresponds to voxels
                        # prediction[self.decoder_id]["pred_masks"][bid] -> numsegs (3076) x 100
                        # target_low_res[bid]["point2segment"] -> 107k -> 0-3075 segid for each point
                        masks = (
                            prediction[self.decoder_id]["pred_masks"][bid]
                            .detach()
                            .cpu()[target_low_res[bid]["point2segment"].cpu()]
                        )
                    voxel_masks_orig = masks.clone()
                    
                # dont do this
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )
                # false, dont do this
                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[
                        offset_coords_idx : curr_coords_idx + offset_coords_idx
                    ]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(
                                        original_pred_masks
                                        * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )

                    scores, masks, classes, heatmap, assignment_top = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                        assignment[bid]
                    )
                # do this, get final scores, masks, classes..
                else:
                    # NOTE: this changes the order of the masks as it picks the topk!!
                    scores, masks, classes, heatmap, assignment_topk = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        # actual number of classes for predictions
                        self.model.num_classes - 1,
                        assignment=assignment[bid]
                    )

                # segments to voxels, voxels to orig points
                # masks: nvoxels, nqueries
                # imap: npoints, 0 to nvoxels
                # p2s: npoints, 0 to nsegments
                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                # in the original mask order
                voxel_masks_orig = (voxel_masks_orig > 0).float()
                fullres_masks_orig = self.get_full_res_mask(
                    voxel_masks_orig,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                if backbone_features is not None:
                    # NOTE: batch size needs to be 1 for this to work, during eval??
                    # get pca features (RGB) onto full res points
                    backbone_features = self.get_full_res_mask(
                        # 122428,3
                        torch.from_numpy(backbone_features),
                        # 299756
                        inverse_maps[bid],
                        # 299756
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            # keep original unsorted pred masks, in the same order as queries
            all_unsorted_pred_masks.append(fullres_masks_orig.numpy())

            # NOTE: changes the order of masks!
            # sort again by scores
            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            # reorder the assignment again according to the score sorting -> has dummy entries for unassigned queries
            # hence same size as masks
            assignment_topk = [assignment_topk[0][sort_scores_index], assignment_topk[1][sort_scores_index]]

            # default false, dont do this
            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not (
                        sort_scores_values[instance_id]
                        < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                # default false -> dont do this 
                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                # do this
                # final predictions to be evaluated
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)
                all_assignment_topk.append(assignment_topk)


        if self.validation_dataset.dataset_name == "scannet200":
            all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
            if self.config.data.test_mode != "test":
                target_full_res[bid]["labels"][
                    target_full_res[bid]["labels"] == 0
                ] = -1

        # get gt and pred boxes
        # bid = sample index within the batch
        # and captions, if they exist
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            # add back label offset to go from pred classes to original
            all_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                ############ STORE PRED BBOX ############
                bbox_data = []

                # NOTE: these masks are SORTED, not in the same order as the queries!!
                for query_id in range(
                    all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        # masks=npoints,nqueries
                        all_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    # keep only the boxes with points in them! 
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        this_bbox_data = (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        bbox_data.append(this_bbox_data)
                # store bbox preds for each sample
                self.bbox_preds[file_names[bid]] = bbox_data

                ######## STORED UNSORTED PRED BBOX FOR CAPTION EVAL #######
                all_unsorted_pred_bbox_data = []

                for query_id in range(
                    all_unsorted_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        # masks=npoints,nqueries
                        all_unsorted_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    # keep only the boxes with points in them! 
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        this_bbox_data = (
                                # TODO: pred class and pred scores should be UNSORTED! but these arent used for caption eval, doesnt matter
                                # keep only the bbox
                                all_pred_classes[bid][query_id].item(), 
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        bbox_data.append(this_bbox_data)
                        # store all bboxes here for caption eval
                        all_unsorted_pred_bbox_data.append(this_bbox_data)
                    else:
                        # store dummy with zeros in all bbox data
                        dummy_bbox_data = (
                            all_pred_classes[bid][query_id].item(),
                            np.zeros(6),
                            all_pred_scores[bid][query_id],
                        )
                        all_unsorted_pred_bbox_data.append(dummy_bbox_data)
                
                # store all the pred bboxes to check for IOU threshold later (c@0.5 evaluation)
                self.all_unsorted_pred_boxes[file_names[bid]] = all_unsorted_pred_bbox_data

                ############ STORE GT BBOX ############
                # GT BOX, get from full res target, not voxels
                gt_bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue
                    
                    # original PC coords
                    obj_coords = full_res_coords[bid][
                        # masks=nobj,npoints
                        target_full_res[bid]["masks"][obj_id, :]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(bool),
                        :,
                    ]
                    # NOTE: this should always be the case for GT objects?
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        gt_bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )
                # store preds/gt bboxes for each sample
                self.bbox_gt[file_names[bid]] = gt_bbox_data
                ######### BBOX PREDS AND GT DONE ##########

                ######### STORE ALL CAPTION RELATED STUFF ##########
                self.assignment_preds[file_names[bid]] = assignment[bid]

                # store gt obj ids and sem IDs
                self.gt_obj_ids[file_names[bid]] = target_low_res[bid]['inst_ids']
                self.gt_sem_ids[file_names[bid]] = target_low_res[bid]['labels']
                # store all original query class predictions
                self.pred_classes_orig[file_names[bid]] = prediction[self.decoder_id]["pred_logits"][bid].argmax(dim=-1)

                if caption_output is not None:
                    # store all the caption preds here
                    self.caption_preds[file_names[bid]] = caption_output[bid]
                    self.caption_gt[file_names[bid]] = cap_gt['cap_gt_corpus'][bid]
                    self.caption_obj_ids[file_names[bid]] = cap_gt['cap_obj_ids'][bid]

                if part_caption_output is not None:
                    self.part_caption_preds[file_names[bid]] = part_caption_output[bid]
                    self.part_caption_gt[file_names[bid]] = cap_gt['part_cap_gt_corpus'][bid]
                    self.part_caption_obj_ids[file_names[bid]] = cap_gt['part_cap_obj_ids'][bid]
                ############# DONE STORING EVERYTHING ############

            # store preds/gt for each sample
            if self.config.general.eval_inner_core == -1:
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }
            else:
                # prev val_dataset
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid][
                        self.validation_dataset.data[idx[bid]]["cond_inner"]
                    ],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }

            if self.config.general.save_visualizations:
                if "cond_inner" in self.validation_dataset.data[idx[bid]]:
                    target_full_res[bid]["masks"] = target_full_res[bid][
                        "masks"
                    ][:, self.validation_dataset.data[idx[bid]]["cond_inner"]]
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid][
                            self.validation_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid][
                            self.validation_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        original_normals[bid][
                            self.validation_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[
                            all_heatmaps[bid][
                                self.validation_dataset.data[idx[bid]]["cond_inner"]
                            ]
                        ],
                        query_pos=all_query_pos[bid][
                            self.validation_dataset.data[idx[bid]]["cond_inner"]
                        ]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features[
                            self.validation_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        point_size=self.config.general.visualization_point_size,
                    )
                else:
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid],
                        original_normals[bid],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[all_heatmaps[bid]],
                        query_pos=all_query_pos[bid]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features,
                        point_size=self.config.general.visualization_point_size,
                    )

            # TODO: save pred-> GT obj ID assignment
            if self.config.general.export:
                if self.validation_dataset.dataset_name == "stpls3d":
                    scan_id, _, _, crop_id = file_names[bid].split("_")
                    crop_id = int(crop_id.replace(".txt", ""))
                    file_name = (
                        f"{scan_id}_points_GTv3_0{crop_id}_inst_nostuff"
                    )

                    self.export(
                        self.preds[file_names[bid]]["pred_masks"],
                        self.preds[file_names[bid]]["pred_scores"],
                        self.preds[file_names[bid]]["pred_classes"],
                        file_name,
                        self.decoder_id,
                    )
                else:
                    ious = []
                    # for 100 queries <-> GT -> ignore the ones where GT is -1
                    all_query_ndx, all_gt_ndx = all_assignment_topk[bid]
                    for query_ndx, gt_ndx in zip(all_query_ndx, all_gt_ndx):
                        if gt_ndx.item() == -1:
                            ious.append(-1)
                            continue
                        query_box = self.all_unsorted_pred_boxes[file_names[bid]][query_ndx.item()][1]
                        gt_box = self.bbox_gt[file_names[bid]][gt_ndx.item()][1]
                        iou = calc_iou(query_box, gt_box)
                        ious.append(iou)
                    # pred boxes -> all unsorted boxes
                    # gt boxes -> for each gt obj, according to gt obj ndx
                    self.export(
                        self.preds[file_names[bid]]["pred_masks"],
                        self.preds[file_names[bid]]["pred_scores"],
                        self.preds[file_names[bid]]["pred_classes"],
                        file_names[bid],
                        self.decoder_id,
                        assignment=all_assignment_topk[bid],
                        obj_ids_orig=target_low_res[bid]['inst_ids'],
                        ious=ious
                    )

    def eval_instance_epoch_end(self, log_prefix="val"):
        ap_results = {}

        head_results, tail_results, common_results = [], [], []

        # pred/gt box is (class, bbox, score)
        # bbox = (center xyz, axis lengths xyz = max-min of points in the box)
        # not stored for all the queries -> only boxes with some points in them are kept
        # need to know the original query index of the box

        # output: recall, precision, AP (over all boxes), over all GT boxes
        box_ap_50 = eval_det(
            self.bbox_preds, self.bbox_gt, ovthresh=0.5, use_07_metric=False,
        )
        # TODO: Store box@50 assignment to be used for caption assignment 
        # create a copy -> assignment_box_at_50
        # use this during caption evaluation
        box_ap_25 = eval_det(
            self.bbox_preds, self.bbox_gt, ovthresh=0.25, use_07_metric=False
        )
        box_ap_25_vals = [v for k, v in box_ap_25[-1].items()]
        mean_box_ap_25 = np.nanmean(box_ap_25_vals)

        box_ap_50_vals = [v for k, v in box_ap_50[-1].items()]
        mean_box_ap_50 = np.nanmean(box_ap_50_vals)

        ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50

        for class_id in box_ap_50[-1].keys():
            class_name = self.validation_dataset.label_info[class_id]["name"]

            key = f"{log_prefix}_{class_name}_val_box_ap_50"
            val = box_ap_50[-1][class_id]

            if not math.isnan(val):
                ap_results[key] = val

        for class_id in box_ap_25[-1].keys():
            class_name = self.validation_dataset.label_info[class_id]["name"]

            key = f"{log_prefix}_{class_name}_val_box_ap_25"
            val = box_ap_25[-1][class_id]

            if not math.isnan(val):
                ap_results[key] = val

        # save to same dir as ckpt
        root_path = self.config.general.save_dir
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_ep={self.current_epoch}_step={self.global_step}_{log_prefix}"

        if self.validation_dataset.dataset_name in [
            "scannet",
            "stpls3d",
            "scannet200",
            "scannetpp"
        ]:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
        else:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"

        # dont mixup train and val eval
        pred_path = f"{base_path}/{log_prefix}_eval.txt"

        if not os.path.exists(base_path) and not self.config.general.no_output:
            os.makedirs(base_path)

        if self.validation_dataset.dataset_name == "s3dis":
            new_preds = {}
            for key in self.preds.keys():
                new_preds[
                    key.replace(f"Area_{self.config.general.area}_", "")
                ] = {
                    "pred_classes": self.preds[key]["pred_classes"] + 1,
                    "pred_masks": self.preds[key]["pred_masks"],
                    "pred_scores": self.preds[key]["pred_scores"],
                }
            mprec, mrec = evaluate(
                new_preds, gt_data_path, pred_path, dataset="s3dis"
            )
            ap_results[f"{log_prefix}_mean_precision"] = mprec
            ap_results[f"{log_prefix}_mean_recall"] = mrec
        elif self.validation_dataset.dataset_name == "stpls3d":
            new_preds = {}
            for key in self.preds.keys():
                new_preds[key.replace(".txt", "")] = {
                    "pred_classes": self.preds[key]["pred_classes"],
                    "pred_masks": self.preds[key]["pred_masks"],
                    "pred_scores": self.preds[key]["pred_scores"],
                }

            _ = evaluate(new_preds, gt_data_path, pred_path, dataset="stpls3d")
        else:
            # pass the validation dataset and get the data from the sample
            # sem*1000 + inst + 1 
            # to be consistent, if the data was changed after loading
            eval_dataset = self.validation_dataset if log_prefix == 'val' else self.train_for_eval_dataset
            print(f'Using dataset for evaluation: {log_prefix} with {len(eval_dataset)} samples' )
            # for scannet, dataset.scene_ids = "0000_00" and pred key = "scene0000_00"
            preds_to_eval = {k: v for (k, v) in self.preds.items() if k in eval_dataset.scene_ids}
            print(f'Kept {len(preds_to_eval)} preds for eval')
            # preds contains all the preds, keep only the ones in this
            avgs_output = evaluate(
                preds_to_eval,
                gt_data_path,
                pred_path,
                dataset_name=self.validation_dataset.dataset_name,
                dataset=eval_dataset,
                eval_each_scene=self.config.general.eval_each_scene,
                no_output=self.config.general.no_output
            )
        for line in avgs_output:
            class_name, _, ap, ap_50, ap_25 = line

            if self.validation_dataset.dataset_name == "scannet200":
                if class_name in VALID_CLASS_IDS_200_VALIDATION:
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap"
                    ] = float(ap)
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap_50"
                    ] = float(ap_50)
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap_25"
                    ] = float(ap_25)

                    if class_name in HEAD_CATS_SCANNET_200:
                        head_results.append(
                            np.array(
                                (float(ap), float(ap_50), float(ap_25))
                            )
                        )
                    elif class_name in COMMON_CATS_SCANNET_200:
                        common_results.append(
                            np.array(
                                (float(ap), float(ap_50), float(ap_25))
                            )
                        )
                    elif class_name in TAIL_CATS_SCANNET_200:
                        tail_results.append(
                            np.array(
                                (float(ap), float(ap_50), float(ap_25))
                            )
                        )
                    else:
                        assert (False, "class not known!")
            else:
                if not math.isnan(float(ap)):
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap"
                    ] = float(ap)
                if not math.isnan(float(ap_50)):
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap_50"
                    ] = float(ap_50)
                if not math.isnan(float(ap_25)):
                    ap_results[
                        f"{log_prefix}_{class_name}_val_ap_25"
                    ] = float(ap_25)

        if self.validation_dataset.dataset_name == "scannet200":
            head_results = np.stack(head_results)
            common_results = np.stack(common_results)
            tail_results = np.stack(tail_results)

            mean_tail_results = np.nanmean(tail_results, axis=0)
            mean_common_results = np.nanmean(common_results, axis=0)
            mean_head_results = np.nanmean(head_results, axis=0)

            ap_results[
                f"{log_prefix}_mean_tail_ap_25"
            ] = mean_tail_results[0]
            ap_results[
                f"{log_prefix}_mean_common_ap_25"
            ] = mean_common_results[0]
            ap_results[
                f"{log_prefix}_mean_head_ap_25"
            ] = mean_head_results[0]

            ap_results[
                f"{log_prefix}_mean_tail_ap_50"
            ] = mean_tail_results[1]
            ap_results[
                f"{log_prefix}_mean_common_ap_50"
            ] = mean_common_results[1]
            ap_results[
                f"{log_prefix}_mean_head_ap_50"
            ] = mean_head_results[1]

            ap_results[
                f"{log_prefix}_mean_tail_ap_25"
            ] = mean_tail_results[2]
            ap_results[
                f"{log_prefix}_mean_common_ap_25"
            ] = mean_common_results[2]
            ap_results[
                f"{log_prefix}_mean_head_ap_25"
            ] = mean_head_results[2]

            overall_ap_results = np.nanmean(
                np.vstack((head_results, common_results, tail_results)),
                axis=0,
            )

            ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
            ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
            ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]
        
            ap_results = {
                key: 0.0 if math.isnan(score) else score
                for key, score in ap_results.items()
            }
        else:
            # get mean AP
            # use np.nanmean instead of statistics.mean to ignore nan values, write in a single line
            mean_ap = np.nanmean([item for key, item in ap_results.items() if key.endswith("val_ap")])
            mean_ap_50 = np.nanmean([item for key, item in ap_results.items() if key.endswith("val_ap_50")])
            mean_ap_25 = np.nanmean([item for key, item in ap_results.items() if key.endswith("val_ap_25")])

            # these shouldnt be Nan -> atleast 1 GT class! could still be nan, np.nanmean(nan) = nan, log as nan
            ap_results[f"{log_prefix}_mean_ap"] = mean_ap
            ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
            ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

        # save all caption preds to file as json
        if self.config.general.gen_captions:
            # eval only on the current set (train/val) captions!! saved predictions contains all captions
            cap_eval_scene_ids = set(self.caption_preds.keys()).intersection(set(eval_dataset.scene_ids))
            caption_scores = self.eval_caps(log_prefix, cap_eval_scene_ids, base_path, eval_dataset, ap_results)

            for caption_metric, score in caption_scores.items():
                ap_results[f'{log_prefix}_{caption_metric}'] = score

        # repeat for part caps, save to different file with part outputs
        if self.config.general.gen_part_captions:
            # eval only on the current set (train/val) captions!! saved predictions contains all captions
            cap_eval_scene_ids = set(self.part_caption_preds.keys()).intersection(set(eval_dataset.scene_ids))
            part_caption_scores = self.eval_caps(log_prefix, cap_eval_scene_ids, base_path, eval_dataset, ap_results, prefix='part_')

            for caption_metric, score in part_caption_scores.items():
                ap_results[f'{log_prefix}_part_{caption_metric}'] = score

        self.log_dict(ap_results)

    def eval_caps(self, log_prefix, cap_eval_scene_ids, base_path, eval_dataset, ap_results, prefix=''):
        '''
        run on either object or part caption preds+gt

        prefix: empty -> object, otherwise part_
        '''
        print(f'Eval caption on {log_prefix} set, {len(cap_eval_scene_ids)} scenes')

        # all preds and gt
        cap_out = defaultdict(dict)
        for scene_id in cap_eval_scene_ids:
            cap_out[scene_id]['preds'] = getattr(self, f'{prefix}caption_preds')[scene_id]
            cap_out[scene_id]['gt'] = getattr(self, f'{prefix}caption_gt')[scene_id]

        cap_out_path = Path(base_path) / f'{log_prefix}_{prefix}all_caption_preds.json'
        if not self.config.general.no_output:
            with open(cap_out_path, 'w') as f:
                json.dump(cap_out, f, indent=4)

        # use assignments to match captions 
        assigned_captions = self.assign_caption_preds(cap_eval_scene_ids, eval_dataset, prefix=prefix)

        # eval captions for matched preds 
        caption_scores, assigned_captions_with_scores, eval_dict = eval_assigned_captions(assigned_captions)

        # plot per class performance
        if self.config.data.instance_classes_file and not self.config.general.no_output:
            class_list = read_txt_list(self.config.data.instance_classes_file)
            plot_cap_eval(assigned_captions_with_scores, eval_dict, Path(base_path) / f'{prefix}caption_eval_plots', class_list)

        # add instance scores, convert to floats as well
        eval_dict['inst_scores'] = {k: float(v) for k, v in ap_results.items()}

        # write captions+scores to file
        caption_output = {
            'preds': assigned_captions_with_scores,
            'eval': eval_dict
        }

        assigned_cap_out_path = Path(base_path) / f'{log_prefix}_{prefix}assigned_caption_preds.json'
        if not self.config.general.no_output:
            with open(assigned_cap_out_path, 'w') as f:
                print('Writing assigned captions to:', assigned_cap_out_path)
                json.dump(caption_output, f, indent=4)
            
        print(f'{prefix}Caption scores:', caption_scores)

        return caption_scores
        

    def assign_caption_preds(self, scene_ids, eval_dataset, prefix):
        assigned_captions = defaultdict(dict)

        # use caption preds, gt and assignments to match them
        for scene_id in scene_ids:
            # gt_indices -> indexes into objs_with_caps (caption gt)
            query_indices, gt_indices = self.assignment_preds[scene_id]
            # for each object with caption
            for obj_id, obj_gt_caption in zip(getattr(self, f'{prefix}caption_obj_ids')[scene_id], getattr(self, f'{prefix}caption_gt')[scene_id]):
                # get the index of this objid in the batch (all objects)
                obj_ndx_in_batch = (self.gt_obj_ids[scene_id] == obj_id).nonzero()
                # skip if not found 
                if obj_ndx_in_batch.shape[0] == 0:
                    continue
                obj_ndx_in_batch = obj_ndx_in_batch.item()
                # get the corresponding query
                query_ndx = query_indices[gt_indices == obj_ndx_in_batch]

                # if this is empty, object has a caption but no assignment -> possible?
                if query_ndx.shape[0] == 0:
                    # everything needed for a prediction 
                    obj_pred_caption = ''
                    save_query_ndx = -1
                    pred_sem_label = 'unknown'
                    pred_bbox = None
                    gt_bbox = None
                    iou = 0
                else:
                    # filter by box AP 0.5
                    # stored bbox is (class, box, score) -> get the box
                    # nqueries boxes for each query
                    pred_bbox = self.all_unsorted_pred_boxes[scene_id][query_ndx][1]
                    gt_bbox = self.bbox_gt[scene_id][obj_ndx_in_batch][1]
                    iou = calc_iou(pred_bbox, gt_bbox)
                    # convert to list for storing in json
                    gt_bbox = gt_bbox.tolist()
                    pred_bbox = pred_bbox.tolist()
                    # TODO: just save the IOU and eval 0.5, 0.25 separately during eval!
                    # get the box IOU
                    if iou < self.config.general.cap_eval_iou_thresh:  # default 0.5
                        obj_pred_caption = ''
                        save_query_ndx = -1
                        pred_sem_label = 'unknown'
                    else:
                        
                        # get the corresponding pred caption
                        obj_pred_caption = getattr(self, f'{prefix}caption_preds')[scene_id][query_ndx]
                        save_query_ndx = query_ndx.item()
                        # get the predicted sem label
                        pred_sem_id = self.pred_classes_orig[scene_id][query_ndx]
                        # map back to orig sem id
                        pred_sem_id_orig = eval_dataset._remap_model_output([pred_sem_id.item()])
                        # get the label
                        pred_sem_label = eval_dataset.label_info[pred_sem_id_orig.item()]['name']

                # get the gt sem label for this obj id
                sem_id = self.gt_sem_ids[scene_id][obj_ndx_in_batch]
                # map to original labels
                gt_orig_sem_id = eval_dataset._remap_model_output([sem_id.item()])
                gt_sem_label = eval_dataset.label_info[gt_orig_sem_id.item()]['name']

                # add to assigned_captions
                assigned_captions[scene_id][int(obj_id)] = {
                    'query_ndx': save_query_ndx,
                    'gt': obj_gt_caption if type(obj_gt_caption) == list else [obj_gt_caption], # convert to list if single GT
                    'pred': obj_pred_caption,
                    'gt_sem_label': gt_sem_label,
                    'pred_sem_label': pred_sem_label,
                    'gt_bbox': gt_bbox,
                    'pred_bbox': pred_bbox,   
                    'box_iou': iou
                }
        return assigned_captions

    def test_epoch_end(self, outputs):
        # list of dicts -> 1 dataset
        if type(outputs[0]) == dict:
            # convert into same format
            outputs = [outputs]
        else:
            # len(outputs) > 1 -> multiple datasets
            # list of list of dicts
            # nothing to do 
            pass

        # multiple datasets, go over each of them
        for eval_ds_name, ds_output in zip(self.eval_datasets, outputs):
            print('Running instance eval on dataset:', eval_ds_name)
            self.eval_instance_epoch_end(log_prefix=eval_ds_name)

            if 'train' in eval_ds_name:
                print('Finish eval on train ds, dont log losses')
                continue

            dd = defaultdict(list)
            # log anything else for val 
            for batch_output in ds_output:
                # each batch output is a dict
                for key, val in batch_output.items():  # .items() in Python 3.
                    dd[key].append(val)

            if self.config.general.eval_against_target:
                print('Evaluating against target, dont log instance losses')
                # caption losses stored, log and continue
                self.log_dict(dd)
                continue


            # mean over all batches
            dd = {k: statistics.mean(v) for k, v in dd.items()}
            dd["val_mean_loss_ce"] = statistics.mean(
                [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
            )
            dd["val_mean_loss_mask"] = statistics.mean(
                [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
            )
            dd["val_mean_loss_dice"] = statistics.mean(
                [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
            )
            # mean of everything
            dd["val_mean_loss"] = statistics.mean(
                [item for item in [v for k, v in dd.items() if "loss" in k]]
            )
            self.log_dict(dd)

        self.clear_preds_data()

        # reinit empty dicts to store preds and GT
        self.init_pred_gt_dicts()

    def clear_preds_data(self):
        # clear all the preds data
        print('Clearing all preds data')
        del self.preds
        del self.bbox_preds
        del self.bbox_gt

        del self.assignment_preds
        del self.all_unsorted_pred_boxes

        del self.caption_preds
        del self.caption_gt
        del self.caption_obj_ids

        del self.part_caption_preds
        del self.part_caption_gt
        del self.part_caption_obj_ids

        del self.gt_obj_ids
        del self.gt_sem_ids
        del self.pred_classes_orig
        gc.collect()
        
    def init_pred_gt_dicts(self):
        # reinit everything
        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.assignment_preds = dict()
        self.all_unsorted_pred_boxes = {}

        self.caption_preds = dict()
        self.caption_gt = dict()
        self.caption_obj_ids = dict()

        self.part_caption_preds = dict()
        self.part_caption_gt = dict()
        self.part_caption_obj_ids = dict()

        # gt obj ids in each scene/sample - can be matched with preds using assignment
        self.gt_obj_ids = dict()
        # corresp sem ids
        self.gt_sem_ids = dict()
        # original predicted classes for each query (from N queries)
        self.pred_classes_orig = dict()

    def configure_optimizers(self):
        optimizers, schedulers = [], []

        # initially use all params in a single group
        model_params_or_group = self.parameters()

        if self.config.general.pretrained_params_lr:
            # all the params in the model
            all_named_params = self.named_parameters()
            # get the list of params in the pretrained model
            # keep the ones that were loaded from ckpt earlier, names in cfg.general.pretrained_params
            # eg. segmentation params
            pretrained_named_params = {name: param for (name, param) in all_named_params if name in self.config.general.pretrained_params}
            pretrained_params = list(pretrained_named_params.values())
            # set the main model params to be the rest of the params
            #eg: caption model params
            main_model_params = [param for name, param in self.named_parameters() if name not in pretrained_named_params]

            model_params_or_group = [
                {'params': pretrained_params, 'lr': self.config.general.pretrained_params_lr}, # stays fixed, atleast with cosineLR
                {'params': main_model_params} # set by the scheduler
            ]

        # optimizer and scheduler for the main model being finetuned/from scratch
        # regular optimizer for rest of the params
        main_optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=model_params_or_group
        )
        optimizers.append(main_optimizer)

        if 'scheduler' in self.config:
            train_loader_len = len(self.train_dataloader())
            if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
                self.config.scheduler.scheduler.steps_per_epoch = train_loader_len
            # cosine scheduler, tmax = epochs * steps_per_epoch
            if "T_max" in self.config.scheduler.scheduler.keys():
                # CosineAnnealingLR
                if self.config.scheduler.scheduler.T_max is not None:
                    print(f'>>> Scheduler T_max was already set to {self.config.scheduler.scheduler.T_max}')
                else:
                    print(f'>>>>>> Set scheduler T_max to {self.config.scheduler.scheduler.T_max}')
                    self.config.scheduler.scheduler.T_max = self.config.trainer.max_epochs * train_loader_len
            if "max_epochs" in self.config.scheduler.scheduler.keys():
                # LinearWarmupCosineAnnealingLR
                self.config.scheduler.scheduler.max_epochs = self.config.trainer.max_epochs * train_loader_len

            lr_scheduler = hydra.utils.instantiate(
                self.config.scheduler.scheduler, optimizer=main_optimizer
            )

            if self.config.general.warmup_steps is not None:
                # wrap scheduler 
                lr_scheduler = WarmupLR(lr_scheduler, init_lr=0.00, num_warmup=self.config.general.warmup_steps, 
                                        warmup_strategy='linear')

            scheduler_config = {"scheduler": lr_scheduler}
            scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
            schedulers.append(scheduler_config)

        return optimizers, schedulers

    def print_sample_stats(self, sample, dataset):
        if not self.config.general.show_sample_stats:
            return
        print('Scene ID:', sample[3])
        unique_labels = np.unique(sample[2][:, 0])
        # semantic class IDs after mapping to 0-N
        unique_labels = unique_labels[unique_labels != self.ignore_label]
        print('Unique labels in sample:', unique_labels)

        # map back to original
        unique_labels_orig = dataset._remap_model_output(unique_labels)
        classes = [self.labels_info[i]['name'] for i in unique_labels_orig if i in self.labels_info]
        print('Unique orig labels in sample:', len(unique_labels_orig), unique_labels_orig, classes)

        # num points
        print('Num points:', sample[0].shape[0])

        # num instances
        unique_instance_ids = np.unique(sample[2][:, 1])
        # remove -100
        unique_instance_ids = unique_instance_ids[unique_instance_ids != self.ignore_label]
        print('Unique instance IDs:', unique_instance_ids)
        print('Num instances:', len(unique_instance_ids))
        # get the names of each instance
        instance_names = []
        instance_sem_labels = []
        for inst_id in unique_instance_ids:
            inst_sem_label = sample[2][sample[2][:, 1] == inst_id, 0][0]
            if inst_sem_label == dataset.ignore_label:
                continue
            instance_sem_labels.append(inst_sem_label)
            inst_sem_label_orig = dataset._remap_model_output(inst_sem_label)
            # labels_info is empty?
            inst_sem_name = self.labels_info[inst_sem_label_orig.item()]['name']
            instance_names.append(inst_sem_name)
        print('Sem labels of samples:', instance_sem_labels)
        print('Instance names:', instance_names)
        # points in instances
        npoints_in_instances = 0
        for inst_id in unique_instance_ids:
            npoints_in_instances += np.sum(sample[2][:, 1] == inst_id)
        print('Num points in instances:', npoints_in_instances)

        # get the number of unique segments
        seg_ids = sample[2][:, 2]
        print('Unique segments:', len(np.unique(seg_ids)))

    def train_sample_stats(self):
        if not self.config.general.show_sample_stats:
            return 
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # orig_sem*1000 + inst + 1
        # gt_data = dataset._remap_model_output(sample[2][:, 0]) * 1000 + sample[2][:, 1] + 1
        train_loader = self.train_dataloader()
        first_batch = next(iter(train_loader))
        # count voxels in each class
        # NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords), target,
            # [sample[3] for sample in batch]
        voxelized_first_sample_sem_labels = first_batch[1][0]['labels']
        print('Voxelized train first sample sem labels:', voxelized_first_sample_sem_labels)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def prepare_data(self):
        # sample: 0:coordinates, 1:features, 2:labels: (sem, inst, segment), 3:sceneid
                # 4:raw_color, 5:raw_normals, 6:raw_coordinates, 7:idx

        # dont create train dataset if not training or evaluating on train
        if self.config.general.train_mode:
            print('****************************************************************')
            print('Creating train dataset..')
            self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
            self.labels_info = self.train_dataset.label_info
            print('Train dataset:', len(self.train_dataset))
            train_sample = self.train_dataset[0]
            self.print_sample_stats(train_sample, self.train_dataset)
            self.train_sample_stats()
        
        print('****************************************************************')
        print('Creating val dataset..')
        self.validation_dataset = hydra.utils.instantiate(self.config.data.validation_dataset)
        print('Val dataset:', len(self.validation_dataset))
        
        if not self.config.general.train_mode:
            # should have atleast a val set if not training?
            self.labels_info = self.validation_dataset.label_info

        if not self.config.general.dont_eval_on_val:
            val_sample = self.validation_dataset[0]
            self.print_sample_stats(val_sample, self.validation_dataset)


        print('****************************************************************')
        # test = val
        print('Setting test set to same as val set')
        self.test_dataset = self.validation_dataset
        print('Test dataset:', len(self.test_dataset))

        # set here, same used for test
        self.eval_datasets = ['val']

        if self.config.general.eval_on_train:
            print('****************************************************************')
            if self.config.general.dont_eval_on_val:
                print('Eval only on train, not val')
                self.eval_datasets = ['train']
            else:
                print('Eval on train and val')
                self.eval_datasets = ['train', 'val']

            n_samples = self.config.general.eval_on_train
            # create a version of the train dataset with mode=val to disable augmentation
            print('Creating train for eval dataset..')
            self.train_for_eval_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
            # actual #samples to keep
            n_keep_samples = min(len(self.train_for_eval_dataset), n_samples) 
            print('Keeping train samples for eval:', n_keep_samples)
            # seed and always keep the same samples
            if self.config.general.eval_on_train_selection == 'random':
                rng = np.random.default_rng(seed=42)
                sample_ndx = rng.choice(len(self.train_for_eval_dataset), n_keep_samples, replace=False)
            elif self.config.general.eval_on_train_selection == 'first_n':
                sample_ndx = [i for i in range(n_keep_samples)]
            new_data = [self.train_for_eval_dataset.data[i] for i in sample_ndx]
            self.train_for_eval_dataset._data = new_data

            print('Train for eval dataset:', len(self.train_for_eval_dataset))

            # set mode to val
            print('Setting train for eval dataset to val mode')
            self.train_for_eval_dataset.mode = 'val'

        if self.config.general.dbg_save_samples:
            # save first train batch and corresp samples
            self.save_samples(self.train_dataset, self.train_dataloader(), 'train')
            self.save_samples(self.validation_dataset, self.val_dataloader(), 'validation')
            # exit
            import sys
            print('****************************************************************')
            print('Exiting after saving samples')
            sys.exit()

        print('Done preparing datasets')
        print('****************************************************************')

    def save_samples(self, dataset, dataloader, save_key):
        # train batch: 
        # NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords), target,
            # [sample[3] for sample in batch]
        # val batch:
        # NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords,
        #       target_full, original_colors, original_normals, original_coordinates, idx), target,
        # [sample[3] for sample in batch] #scene IDs
        save_dir = Path(self.config.general.save_dir) / 'samples'
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving dbg {save_key} samples to:', save_dir)
        batch = next(iter(dataloader))
        torch.save(batch, save_dir / f'{save_key}_batch.pth')

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    # val and log
    # additionally, on the train set (with mode=val to disble augmentation) if enabled
    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        val_loader =  hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

        if self.config.general.eval_on_train:
            train_for_eval_loader = hydra.utils.instantiate(
                self.config.data.validation_dataloader,
                self.train_for_eval_dataset,
                collate_fn=c_fn,
            )

            if self.config.general.dont_eval_on_val:
                # train only
                return [train_for_eval_loader]
            else:    
                # train and val
                return [train_for_eval_loader, val_loader]
        else:
            # val only
            return val_loader

    # test on the val set
    # additionally, on the train set (with mode=val to disble augmentation) if enabled
    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        test_loader = hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

        if self.config.general.eval_on_train:
            train_for_eval_loader = hydra.utils.instantiate(
                self.config.data.validation_dataloader,
                self.train_for_eval_dataset,
                collate_fn=c_fn,
            )
            if self.config.general.dont_eval_on_val:
                # train only
                return [train_for_eval_loader]
            else:    
                # train and test
                return [train_for_eval_loader, test_loader]
        else:
            # test only
            return test_loader
            
