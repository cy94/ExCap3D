import itertools
from peft import LoraModel, LoraConfig, get_peft_model
from models.position_embedding import PositionEmbeddingCoordsSine
import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch_scatter import scatter_mean, scatter_max
from torch import nn, Tensor
from typing import Dict

from collections import OrderedDict
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import  T5Config, T5Tokenizer, T5ForConditionalGeneration


import numpy as np
from models.mask3d_captioner.generation_utils import generation


def position_embedding(max_len: int, d_model: int) -> Tensor:
    position_embedding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)
    return position_embedding


class Mask3D_Captioner(nn.Module):
    def __init__(self, 
            embedding_size,
            max_caption_length,
            query_dim,
            num_beams,
            use_pretrained,
            finetune_layers,
            n_layer,
            n_head,
            use_context_feats,
            context_nn,
            context_dim,
            model_name,
            gt_sem_onehot_as_query,
            pred_sem_onehot_as_query,
            num_classes, # number of target classes for mask3d model, subtract 1 to get actual number of sem classes
            dont_project_queries,
            class_weights_file,
            viz_attn_mask,
            use_obj_segment_feats, obj_feat_dim, obj_feat_type, use_obj_feat_in,
            tokenizer_model_name,
            context_use_pos_enc,
            segment_aggr_type,
            use_lora, lora_rank,
            output_hidden_states,
            output_beam_hidden_states,
            use_other_caption_feats,
            use_hidden_state_ndx, #which hidden state to use for cross attn
            ):
        super(Mask3D_Captioner, self).__init__()

        self.use_hidden_state_ndx = use_hidden_state_ndx

        # use hidden states from other captioner -> gets set externally
        self.use_other_caption_feats = use_other_caption_feats

        self.output_hidden_states = output_hidden_states
        self.output_beam_hidden_states = output_beam_hidden_states

        self.segment_aggr_type = segment_aggr_type
        self.context_use_pos_enc = context_use_pos_enc
        self.use_obj_feat_in = use_obj_feat_in
        self.obj_feat_dim = obj_feat_dim
        self.use_obj_segment_feats = use_obj_segment_feats
        self.obj_feat_type = obj_feat_type
        self.viz_attn_mask = viz_attn_mask

        # set to part_ for part caption model
        self.tgt_prefix = ''

        self.dont_project_queries = dont_project_queries
        self.gt_sem_onehot_as_query = gt_sem_onehot_as_query
        self.pred_sem_onehot_as_query = pred_sem_onehot_as_query
        # if any of these, set query dim to number of classes
        if self.gt_sem_onehot_as_query or self.pred_sem_onehot_as_query:
            print('Use GT semantic as query?:', self.gt_sem_onehot_as_query)
            print('Use pred semantic as query?:', self.pred_sem_onehot_as_query)
            query_dim = num_classes - 1
            print('Setting query dim to number of semantic classes:', query_dim)

        self.num_classes = num_classes - 1

        self.use_context_feats = use_context_feats
        self.context_nn = context_nn

        self.embedding_size = embedding_size
        # max number of tokens -> increase this to get more than 64 outputs
        self.max_positions = 2*max_caption_length

        self.model_name = model_name
        
        # NOTE: should be same as the tokenizer in the semseg dataset
        # initialize tokenizer and set tokenizer's `padding token` to `eos token`
        if 'gpt' in tokenizer_model_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif 't5' in tokenizer_model_name: 
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name)
        
        self.nvocabs = self.tokenizer.vocab_size

        if use_pretrained: # model name should be specified
            if 'gpt' in model_name:
                # emb size is from pretrained model, 768
                print(f'Using pretrained GPT2 model: {model_name}')
                self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
                self.embedding_size = self.transformer.config.n_embd
                print(f'Setting embedding size to pretrained model size: {self.embedding_size}')
            elif 't5' in model_name:
                self.transformer = T5ForConditionalGeneration.from_pretrained(model_name)
                print(f'Using pretrained model: {model_name}')
        else:
            # create a config and then the model
            # use same arch as pretrained or custom? by default use same arch
            if model_name is None: #gpt model according to the specified params
                ## caption generation cores
                gpt2_config = GPT2Config(
                    vocab_size=self.nvocabs,
                    n_positions=self.max_positions,
                    n_embd=self.embedding_size,
                    n_layer=n_layer,
                    n_head=n_head,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    add_cross_attention=True, # always add cross attn
                )
                # can use custom emb size
                print('Using from-scratch GPT2 model with custom arch')
                self.transformer = GPT2LMHeadModel(config=gpt2_config)
            elif 'gpt2' in model_name:
                gpt2_config = GPT2Config.from_pretrained(model_name)    
                gpt2_config.n_positions = self.max_positions # set to same as ours
                gpt2_config.add_cross_attention = True # always add cross attn
                print(f'Setting embedding size to pretrained model size: {self.embedding_size}')
                self.embedding_size = gpt2_config.n_embd
                print(f'Using from-scratch GPT2 model with pretrained arch: {model_name}')
                self.transformer = GPT2LMHeadModel(config=gpt2_config)
            elif 't5' in model_name:
                t5_config = T5Config.from_pretrained(model_name)
                print(f'Using from-scratch T5 model with pretrained arch: {model_name}')
                self.transformer = T5ForConditionalGeneration(config=t5_config)

        if use_lora: # finetune lora params only
            lora_config =  LoraConfig(
                    task_type="CAUSAL_LM",
                    r=lora_rank,
                    lora_alpha=32,
                    target_modules=['c_proj'],
                    lora_dropout=0.01,
                )
            self.transformer = get_peft_model(self.transformer, lora_config) #works
            print('Use LoRA with rank:', lora_rank)
        else:
            finetuned_params = []
            if finetune_layers is not None:
                # freeze all layers
                # NOTE: lm_head isnt in this list, but has requires_grad=True
                for name, param in self.transformer.named_parameters():
                    should_finetune = any([x in name for x in finetune_layers])
                    if not should_finetune:
                        param.requires_grad = False
                    else:
                        finetuned_params.append(name)

            print('Finetuned caption model params:', finetuned_params)
            
        # change the position embedding to max positions x emb size from 1024x768
        if self.model_name is None or 'gpt' in self.model_name:
            self.transformer.transformer.wpe = nn.Embedding.from_pretrained(
                position_embedding(self.max_positions, self.embedding_size)
            )

        if not self.dont_project_queries:
            ## for proposal feature projection
            self.query_projector = nn.Sequential(
                nn.Linear(query_dim, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.ReLU(),
            )

        if self.use_context_feats:
            self.context_projector = nn.Sequential(
                nn.Linear(context_dim, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.ReLU(),
            )

        if self.use_obj_segment_feats:
            self.proj_obj_feats = nn.Sequential(
                nn.Linear(obj_feat_dim, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.ReLU(),
            )

        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': num_beams,
            'max_length': max_caption_length,
        }

        self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=128, # output embedding dim
                gauss_scale=1.0, # same as mask3d model
                normalize=True,
            )

        self.class_weights = None
        if class_weights_file is not None:
            self.class_weights = torch.Tensor(np.loadtxt(class_weights_file))
        
    def forward(self, instseg_output: dict, 
                    assignment: list, 
                    caption_gt: dict,
                    target,
                    is_eval: bool=False,
                ) -> dict:
        
        if is_eval is False:
            caption_output, caption_loss, caption_token_acc, extra_output = self.forward_training(instseg_output, assignment, caption_gt, target)
            return caption_output, caption_loss, caption_token_acc, extra_output
        else:
            with torch.no_grad(): # this shouldnt be needed, nograd + model.eval should be set by lightning
                # do training forward pass to get the loss
                # dont use training caption output (logits, etc), use eval output
                # NOTE: ignore caption output from train
                _, eval_caption_loss, eval_caption_token_acc, extra_output = self.forward_training(instseg_output, assignment, caption_gt, target)
            # eval forward pass (beam search) to get eval preds and extra outputs
            # NOTE: ignore extra_output from eval
            # TODO: need hidden states from val -> 
            eval_caption_output, val_extra_output = self.forward_evaluation(instseg_output, assignment, caption_gt, target)

            if extra_output is None:
                extra_output = {}
                
            if 'val_hidden_states' in val_extra_output:
                # use val hidden states only for inference (shared between caption models)
                extra_output['val_hidden_states'] = val_extra_output['val_hidden_states']
            
            return eval_caption_output, eval_caption_loss, eval_caption_token_acc, extra_output
    
    def pick_queries(self, queries, assignment):
        # queries: bsize, nqueries, 128
        keep_queries  = []
        for _, (sample_queries, sample_assignment) in enumerate(zip(queries, assignment)):
            # which queries to keep, make it a single indexing tensor
            query_ndx = torch.LongTensor(sample_assignment[0])
            # pick the queries
            sample_keep_queries = sample_queries[query_ndx]
            # should be n,dim, if not unsqueeze it
            if sample_keep_queries.ndim == 1:
                sample_keep_queries = sample_keep_queries.unsqueeze(0)

            if sample_keep_queries.shape[0] == 0:
                # no queries to keep
                continue
            keep_queries.append(sample_keep_queries)
        return keep_queries
    
    def get_assignment_existing_objects_old(self, assignment, cap_obj_ids_batch, target_batch):
        '''
        get a new copy of the assignment (query ndx->gt obj ndx)
        so that we can keep only the queries (and other GT properties) for objects that have captions
        update the assignment and remove objects without captions
        note: if all gt objects have captions, output will be same as input

        cap_obj_ids = list of object ids that have captions
        keep assignments only for these

        output: list, for each sample in the batch -
        [query_ndx], [gt_ndx] -> gt_ndx now indexes into cap_obj_ids instead of all the objs in the batch
        '''
        assignment_objs_with_caps = []

        for sample_assignment, sample_cap_obj_ids, sample_target in zip(assignment, cap_obj_ids_batch, target_batch):
            # gt_obj_ndx_all -> index into the objects in the sample (0,1,2)
            query_ndx_all, gt_obj_ndx_all = sample_assignment

            # get the actual object ids = inst_ids from the sample target using gt_obj_ndx_all
            # then check if they are in cap_obj_ids (objs with captions)

            # which of the gt obj ndx to keep?
            # NOTE: keep_gt_ndx indexes into gt_obj_ndx_all
            keep_gt_ndx = [i for i, gt_obj_ndx in enumerate(gt_obj_ndx_all) if sample_target['inst_ids'][gt_obj_ndx] in sample_cap_obj_ids]
            # keep the corresponding query ndx -> indexes into queries
            new_query_ndx = [query_ndx_all[i].item() for i in keep_gt_ndx]

            # keep the gt ndx -> index into caption obj ids where the obj ID matches
            # remap old assignment to new assignment IDs because captionids is smaller than the whole batch
            new_gt_ndx = [sample_cap_obj_ids.index(sample_target['inst_ids'][gt_obj_ndx_all[i]].item()) for i in keep_gt_ndx]

            assignment_objs_with_caps.append((new_query_ndx, new_gt_ndx))
        return assignment_objs_with_caps

    def get_assignment_existing_objects(self, assignment, cap_obj_ids_batch, target_batch):
        assignment_objs_with_caps = []

        for sample_assignment, sample_cap_obj_ids, sample_target in zip(assignment, cap_obj_ids_batch, target_batch):
            # keep only gt_ndx that are in cap_obj_ids and the corresponding query_ndx
            query_ndx_all, gt_obj_ndx_all = sample_assignment

            # for this sample -> each is a subset of the original query ndx and gt ndx
            new_query_ndx, new_gt_ndx = [], []

            for obj_id in sample_cap_obj_ids:
                # get the index on this obj_id in the sample
                ndx_in_gt_objs = sample_target['inst_ids'].tolist().index(obj_id)
                # where is this in the gt part of the assignment
                try:
                    ndx_in_gt_assignment = gt_obj_ndx_all.tolist().index(ndx_in_gt_objs)
                except ValueError:
                    # this object is not in the instance GT?
                    continue
                # corresponding query ndx
                query_ndx = query_ndx_all[ndx_in_gt_assignment].item()

                new_query_ndx.append(query_ndx)
                new_gt_ndx.append(ndx_in_gt_objs)

            assignment_objs_with_caps.append((new_query_ndx, new_gt_ndx))

        return assignment_objs_with_caps

    def pick_caption_gt(self, gt_prop, cap_obj_ids, assignment):
        '''
        Pick part of the GT property according to capobjids (all objIDs that have captions) 

        gt_prop: flat list of gt properties on *objects with captions*, over all scenes, such as objID, semID, sceneID
        NOTE: gt properties are in the same order as cap_obj_ids

        and assignment ->
            assignment[i] = for scene i, (pred assignment, gt_assignment) = (list of pred ndx, list of gt ndx) 
                            such that pred j is assigned to jth obj in cap_obj_ids
            NOTE: assignment has already been filtered for objs that have captions, this isnt the original assignment for instance segmentation
            NOTE: gt_assignment now indexes into cap_obj_ids instead of all the objs in the batch
        '''
        nobjs_done = 0
        picked_gt_prop = []

        for (_, gt_assignment), sample_cap_obj_ids in zip(assignment, cap_obj_ids):
            # num objs that have captions
            n_objs = len(sample_cap_obj_ids)

            # get the GT property for this scene (just need to know the number of objects with captions)
            sample_gt_prop = gt_prop[nobjs_done:nobjs_done+n_objs]

            # pick caption ids in the assignment
            # gt assignment indexes into obj caption ids now, not the original target
            
            # for tensors, direct indexing
            if isinstance(sample_gt_prop, torch.Tensor):
                picked_gt_prop.append(sample_gt_prop[gt_assignment])
            else: # list of scene ids, etc - have pick individual elements by index
                picked_gt_prop.append([sample_gt_prop[i] for i in gt_assignment])

            # total objs seen till now, should add up to len(gt_prop)
            nobjs_done += n_objs

        assert nobjs_done == len(gt_prop), 'Number of objects with captions should match the total number of GT properties to be selected'

        if isinstance(picked_gt_prop[0], torch.Tensor):
            picked_gt_prop = torch.cat(picked_gt_prop)
        else:
            picked_gt_prop = list(itertools.chain(*picked_gt_prop))

        return picked_gt_prop

    def get_queries(self, instseg_output, assignment, target):
        # get the actual queries or fake query features based on semantic GT/preds
        # get queries corresponding to assignments
        # final queries is bsize, nqueries, 128
        bsize = instseg_output['final_queries'].shape[0]
        nqueries = instseg_output['final_queries'].shape[1]
        device = instseg_output['final_queries'].device

        if self.gt_sem_onehot_as_query or self.pred_sem_onehot_as_query:
            # get fake "query features" corresp to the assigned GT objects
            # = one hot of GT semantic classes
            
            if self.dont_project_queries:
                # this is directly the embedding, dont project again
                query_dim = self.embedding_size
            else:
                query_dim = self.num_classes

            queries = torch.zeros(bsize, nqueries, query_dim, device=device)

            if self.gt_sem_onehot_as_query:
                # insert semantic one hot into queries
                for ndx in range(bsize):
                    query_assignment, gt_assignment = assignment[ndx]
                    # get the 
                    # N,
                    all_sem_labels = target[ndx]['labels']
                    assigned_sem_labels = all_sem_labels[gt_assignment]
                    # N, num_classes, make it float to be a "feature"
                    one_hot = torch.nn.functional.one_hot(assigned_sem_labels, self.num_classes).float().to(device)
                    # insert into the places where the original queries were, rest not required
                    # set the first num_classes columns, rest are dummy zeros in case the query is not projected
                    queries[ndx, query_assignment, :self.num_classes] = one_hot
            elif self.pred_sem_onehot_as_query:
                # use predicted semantic classes, remove the last column (bg class)
                # other than last dim, has the same shape as actual queries
                logits = instseg_output['pred_logits'][..., :-1]

                # insert semantic pred logits into zero queries
                queries[:, :, :self.num_classes] = logits
        else:
            # use the actual queries
            queries = instseg_output['final_queries']

        return queries

    def get_context_feats(self, picked_query_locs, all_query_locs, all_feats):
        '''
        picked_query_locs: list of [N, 3]
        all_query_locs: B, nqueries, 3
        all_feats: B, nqueries, 128
        '''
        ctx_feats = []
        ctx_locations = []
        ctx_ndx = []

        k = all_feats.shape[1] if self.context_nn == 'all' else self.context_nn

        for batch_ndx in range(len(picked_query_locs)):
            # find distance between picked queries and all queries
            # and pick the top context_nn in the remaining, if 'all' keep all feats
            picked_query_loc_sample = picked_query_locs[batch_ndx].unsqueeze(0) # add batch dim, 1,nobj,3
            all_query_loc_sample = all_query_locs[batch_ndx].unsqueeze(0) # add batch dim, 1,nquery,3

            # npicked, nqueries
            dist = torch.cdist(picked_query_loc_sample, all_query_loc_sample).squeeze(dim=0) # remove batch dim
            # top by distance npicked, k
            _, topk_ndx = torch.topk(dist, k, largest=False)
            # nobj, k
            ctx_ndx.append(topk_ndx)
            ctx_locations_sample = all_query_loc_sample[0][topk_ndx]
            ctx_locations.append(ctx_locations_sample)

            all_feats_sample = all_feats[batch_ndx] # nquery, 128
            # pick the topk feats
            ctx_feats_sample = all_feats_sample[topk_ndx]
            ctx_feats.append(ctx_feats_sample)

        return ctx_feats, ctx_locations, ctx_ndx

    def get_obj_segment_feats(self, instseg_output, assignment, target):
        out_feats = [] # nobj, nsegments (varies), featdim -> pad to max segments?

        device = instseg_output['features3d'][0].device

        # original mask3d feats / backprojected 2d feats
        bbone_feats = instseg_output['features3d']

        for (pred_masks, sample_target, sample_assignment, sample_point_feats) in zip(instseg_output['pred_masks'], target, assignment, bbone_feats):
            pred_masks = pred_masks.detach().cpu() # nseg, nquery
            segment_masks = pred_masks > 0.5 # nseg, nquery
            # nseg, 96, 0feats if no indices
            if self.segment_aggr_type == 'mean':
                segment_feats = scatter_mean(sample_point_feats, sample_target['point2segment'], dim=0)
            elif self.segment_aggr_type == 'max':
                segment_feats, _ = scatter_max(sample_point_feats, sample_target['point2segment'], dim=0)

            # go through each object and get feats for that object
            for query_ndx in sample_assignment[0]: # 0=query, 1=gt
                # segments in this object
                obj_seg_mask = segment_masks[:, query_ndx] 
                obj_segment_feats = segment_feats[obj_seg_mask]

                if obj_seg_mask.sum() == 0:
                    # create 0 features with dim self.obj_feat_dim
                    obj_feats = torch.zeros(self.obj_feat_dim).to(device).unsqueeze(0) # 1, 96
                elif self.obj_feat_type == 'avg':
                    obj_feats = obj_segment_feats.mean(0).unsqueeze(0) # nseg, 96 -> 1,96
                elif self.obj_feat_type == 'segment':
                    obj_feats = obj_segment_feats
                out_feats.append(obj_feats)

        return out_feats

    def pad_feats(self, feats_list):
        # pad list of N,dim to max_length,dim
        out_feats = []
        max_length = max([x.shape[0] for x in feats_list])
        for feat in feats_list:
            pad_length = max_length - feat.shape[0]
            if pad_length > 0:
                padded_feat = torch.cat([feat, torch.zeros(pad_length, feat.shape[1]).to(feat.device)], dim=0)
            else:
                padded_feat = feat
            out_feats.append(padded_feat)

        return out_feats

    def prepare_context(self, instseg_output, target, assignment_objs_with_caps, inputs, attn_masks, num_prefix_tokens, extra_output, mode):
        context_feats = None
        if self.use_context_feats:
            # for each picked query, create the corresp context feats
            query_coords = torch.Tensor(instseg_output['query_centers']).to(self.device) # B, nqueries, 3
            picked_query_locs = self.pick_queries(query_coords, assignment_objs_with_caps)
            picked_context_feats, picked_context_locations, picked_context_ndx = self.get_context_feats(picked_query_locs, query_coords, instseg_output['final_queries'])
            extra_output['context_ndx'] = [x.detach().cpu().numpy() for x in picked_context_ndx]
            extra_output['query_locs'] = [x.detach().cpu().numpy() for x in picked_query_locs]
            extra_output['context_locs'] = [x.detach().cpu().numpy() for x in picked_context_locations]
            context_feats = torch.cat(picked_context_feats) # numobj,numctx,querydim
            if self.context_use_pos_enc: # add pos enc of context locations
                all_pos_enc = []
                for ndx, ctx_pos in enumerate(picked_context_locations):
                    n_obj = ctx_pos.shape[0]
                    # repeat scene bounds n_obj times
                    scene_bounds = [instseg_output['raw_coord_mins'][ndx].unsqueeze(0).repeat(n_obj, 1), instseg_output['raw_coord_maxs'][ndx].unsqueeze(0).repeat(n_obj, 1)]
                    pos_enc = self.pos_enc(ctx_pos, input_range=scene_bounds)  # nobj,nctx,3 -> nobj,dim,nctx
                    all_pos_enc.append(pos_enc.permute(0, 2, 1)) # nobj,nctx,dim
                # join all together
                all_pos_enc = torch.cat(all_pos_enc, dim=0)
                # concat with context feats
                context_feats = torch.cat([context_feats, all_pos_enc], dim=-1) # nobj,nctx,dim+dim
            # project to embedding dim
            context_feats = self.context_projector(context_feats) # nobj,nctx,embdim
        elif self.use_obj_segment_feats: # use 3d backbone features + any extra features
            context_feats = self.get_obj_segment_feats(instseg_output, assignment_objs_with_caps, target) # avg feats or per segment feats

            if self.obj_feat_type == 'avg':
                context_feats = torch.stack(context_feats) # nobj,1,featdim
                context_feats = self.proj_obj_feats(context_feats) #nobj,1,embdim

                if self.use_obj_feat_in == 'input': # put feats into the input, not context
                    if mode == 'train':
                        context_feats = context_feats.unsqueeze(0) #1,nobj,1,dim
                    num_prefix_tokens += 1 # join with queries along dim 2 
                    inputs.append(context_feats)
                    attn_masks.append(torch.ones_like(context_feats[..., 0]))
                    context_feats = None # make context empty 
                elif self.use_obj_feat_in == 'ctx':
                    pass # nobj,1,dim
            elif self.obj_feat_type == 'segment':
                context_feats = torch.stack(self.pad_feats(context_feats)) # nobj, maxnumseg, dim
                context_feats = self.proj_obj_feats(context_feats) #nobj, maxnumseg, embdim

                if self.use_obj_feat_in == 'input': # put feats into the input, not context
                    maxlen = context_feats.shape[1] # max numsegs
                    num_prefix_tokens += maxlen
                    if mode == 'train':
                        context_feats = context_feats.unsqueeze(0) #1,nobj,maxnumseg,dim
                    inputs.append(context_feats)
                    attn_masks.append(torch.ones_like(context_feats[..., 0]))
                    context_feats = None # make context empty
                elif self.use_obj_feat_in == 'ctx':
                    pass # nothing to be done
        return context_feats, inputs, attn_masks, extra_output, num_prefix_tokens

    def get_other_caption_feats(self, other_caption_outputs, current_scene_ids, current_obj_ids):
        '''
        other_caption_outputs: dict of hidden states, scene_ids, obj_ids from previous caption model
        current_scene_ids: list of scene ids for the current batch
        current_obj_ids: list of obj ids for the current batch

        return: 
        hidden_states: list of hidden states for the current scene ids and obj ids from the previous caption model
        '''
        if other_caption_outputs is None or 'hidden_states' not in other_caption_outputs:
            # might not have got hidden states from the other captioner -> no matched queries
            # len(current_obj_ids), 1, emb dim
            other_matched_hidden_states = torch.zeros(len(current_obj_ids), 1, self.embedding_size)
            return other_matched_hidden_states #send it to the correct device outside this function

        other_hidden_states = other_caption_outputs['hidden_states'][self.use_hidden_state_ndx]
        other_sceneids = other_caption_outputs['scene_ids']
        # convert to list so that we can compare objids as simple integers, not tensors
        other_objids = other_caption_outputs['obj_ids'].tolist()

        other_matched_hidden_states = []

        # get the hidden states for the current scene ids and obj ids if present, otherwise use 0 feats
        # convert obj id tensor to list
        for scene_id, obj_id in zip(current_scene_ids, current_obj_ids.tolist()):
            got_feat = False
            for other_scene_id, other_obj_id, other_hidden_state in zip(other_sceneids, other_objids, other_hidden_states):
                if scene_id == other_scene_id and obj_id == other_obj_id:
                    other_matched_hidden_states.append(other_hidden_state)
                    got_feat = True
            if not got_feat:
                # seqlen, dim
                # assumes that other hidden states has atleast one entry because we dont know the seqlen
                other_matched_hidden_states.append(torch.zeros_like(other_hidden_states[0]))

        other_matched_hidden_states = torch.stack(other_matched_hidden_states)
        return other_matched_hidden_states


    def forward_training(self, instseg_output, assignment, caption_gt, target):
        '''
        NOTE: captioner does not predict <bos> token
        return: caption_output, caption_loss, caption_token_acc, extra_output
        
        caption_output: model output with logits, hidden_states
        extra_output: hidden states, attention weights, objids, semids, sceneids for the objects that we trained on
        '''
        extra_output = {}
        inputs, attn_masks, context_feats = [], [], None # model inputs
        num_prefix_tokens = 1 # use 1 query token, more can be added later
        ########### prepare queries ############
        prefix = self.tgt_prefix # '' or 'part_'
        queries = self.get_queries(instseg_output, assignment, target)

        ####  NOTE: new gt indices are into cap_obj_ids
        assignment_objs_with_caps = self.get_assignment_existing_objects_old(assignment, caption_gt[f'{prefix}cap_obj_ids'], target)

        picked_queries = self.pick_queries(queries, assignment_objs_with_caps)
        # add new dim for bsize=1, ntokens=1
        # handle the case where theres nothing to train
        if len(picked_queries) == 0:
            print('No queries to train on')
            # nothing to match, return
            return None, None, None, None

        # nmatched, 128 - add batch dim and len=1 dim
        # should be b, nobj, len=1, dim=128
        matched_queries = torch.cat(picked_queries)

        if matched_queries.shape[0] == 0:
            print('No queries to train on')
            # nothing to match, return
            return None, None, None, None

        # by default, project queries to llm embedding dim: nobj, embdim
        if not self.dont_project_queries:
            matched_queries = self.query_projector(matched_queries)
        # 1, nobj, 1, embdim
        matched_queries = matched_queries.unsqueeze(0).unsqueeze(2)
        
        inputs.append(matched_queries)
        attn_masks.append(torch.ones_like(matched_queries[..., 0]))
        ############# prepare context  ################
        # this updates the inputs and attn_masks lists and num_prefix_tokens (if the context is in the input sequence)
        context_feats, inputs, attn_masks, extra_output, num_prefix_tokens = self.prepare_context(instseg_output, target, assignment_objs_with_caps, inputs, attn_masks, num_prefix_tokens, extra_output, 'train')
        ############ prepare captions ############
        # pick the GT caption ids into the order in the assignment!
        caption_ids_picked = self.pick_caption_gt(caption_gt[f'{prefix}cap_gt_tokens'], caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps)
        caption_mask_picked = self.pick_caption_gt(caption_gt[f'{prefix}cap_gt_attn_mask'], caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps)

        # objids = one for list for each sample
        # sceneids for each original obj id

        # TODO: this can have a size different from caption_ids_picked?? why?
        scene_ids = list(itertools.chain(*[[scene_id for _ in range(len(caption_gt[f'{prefix}cap_obj_ids'][bid]))] for bid, scene_id in enumerate(instseg_output['scene_ids'])]))
        caption_scene_ids_picked = self.pick_caption_gt(scene_ids, caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps)

        cap_obj_ids_concat = torch.cat([torch.LongTensor(objids) for objids in caption_gt[f'{prefix}cap_obj_ids']])
        caption_obj_id_picked = self.pick_caption_gt(cap_obj_ids_concat, caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps).flatten()
        
        # output from the previous captioner
        # NOTE: only works during training because we have the assignment and objects which have captions
        # during evaluation, hidden states are used only in forward_eval and not here
        if self.training and self.use_other_caption_feats:
            # instseg output['captioner_output] has scene_ids, obj_ids -> match with the current ones and pick the required hidden states
            # nobj_current, seqlen, dim
            other_matched_hidden_states = self.get_other_caption_feats(instseg_output['captioner_output'], caption_scene_ids_picked, caption_obj_id_picked)
            other_matched_hidden_states = other_matched_hidden_states.to(matched_queries.device)
            
            # add this to context feats if it already exists
            if context_feats is not None:
                # nobj, seqlen, dim
                context_feats = torch.cat([context_feats, other_matched_hidden_states], dim=1)
            else:
                context_feats = other_matched_hidden_states

        # sem ids of captioned objects, to get weights
        caption_sem_id_picked = self.pick_caption_gt(caption_gt[f'{prefix}cap_sem_ids'], caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps).flatten()
        caption_mask_picked = caption_mask_picked.unsqueeze(0)
        attn_masks.append(caption_mask_picked)

        if self.model_name is None or 'gpt' in self.model_name:
            caps_emb = self.transformer.transformer.wte(caption_ids_picked).unsqueeze(0) # input embedding
        elif 't5' in self.model_name:
            caps_emb = self.transformer.shared(caption_ids_picked).unsqueeze(0) # dim is 512
        inputs.append(caps_emb)
        ############ finalize inputs and forward ############
        # cat along ntokens dim
        # 1, nobjs, ntokens, 128
        inputs_embeds = torch.cat(inputs, dim=2)  
        inputs_masks = torch.cat(attn_masks, dim=2)   

        caption_output = self.transformer( 
            # numobj x (1 + max_des_len) x querydim
            inputs_embeds=inputs_embeds[0],
            # numobj x (1 + max_des_len)
            attention_mask=inputs_masks[0],
            encoder_hidden_states=context_feats, # 
            output_attentions=self.viz_attn_mask,
            output_hidden_states=self.output_hidden_states
        )
        
        # store extra computed things for later use
        extra_output.update({
            'obj_ids': caption_obj_id_picked,
            'sem_ids': caption_sem_id_picked,
            'scene_ids': caption_scene_ids_picked,
        })

        if self.output_hidden_states:
            extra_output['hidden_states'] = caption_output.hidden_states
            
        # use on val set with bsize1
        if self.viz_attn_mask:
            # obj ids
            caption_obj_id_picked = self.pick_caption_gt(torch.LongTensor(caption_gt[f'{prefix}cap_obj_ids'][0]), caption_gt[f'{prefix}cap_obj_ids'], assignment_objs_with_caps).flatten()
            # get token-3d context cross attn, tuple of 2x (nobj/batch_size, num_heads, caption sequence_length, context sequence_length)
            x_attn = caption_output.cross_attentions
            logits = caption_output.logits[:, :-1]
            pred_tokens = torch.argmax(logits, dim=-1)
            # decode each token separately to viz xattn 
            pred_token_strings = [[self.tokenizer.decode(token) for token in tokens] for tokens in pred_tokens.tolist()]
            # get the text output
            # get text captions
            pred_captions = self.tokenizer.batch_decode(
                pred_tokens.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            extra_output.update({
                'cross_attention': [layer_x_attn.detach().cpu().numpy() for layer_x_attn in x_attn],
                'pred_captions': pred_captions,
                'pred_token_strings': pred_token_strings,
                'obj_ids': caption_obj_id_picked.tolist()
            })

        ############# evaluate ##############
        caption_token_acc = self.compute_token_accuracy(caption_output, caption_ids_picked, num_prefix_tokens)
        # logits: bsize, ntokens+1, vocabsize -> exclude last token prediction (not in GT)
        # get actual predicted tokens
        # caption_ids: bsize, ntoken

        weights = None
        if self.class_weights is not None:
            # get weights according to the semantic IDs of the captioned objects
            # weight for each sample according to its semantic class
            weights = self.class_weights.to(caption_sem_id_picked.device)[caption_sem_id_picked]

        # logit: [b, ntokens+1, vocab=50257]
        caption_loss = self.loss_caption(
            # no supervision for prefix tokens -> numprefix-1
            # -1 -> no supervision for the last logit (extra token predicted)
            logits = caption_output.logits[:, (num_prefix_tokens-1):-1, :],
            target = caption_ids_picked.long(),
            # nobj
            weights = weights
        )

        return caption_output, caption_loss, caption_token_acc, extra_output
    
    def compute_token_accuracy(self, caption_output, gt_caption_ids, num_prefix_tokens):
        logits = caption_output.logits[:, (num_prefix_tokens-1):-1]
        pred_tokens = torch.argmax(logits, dim=-1)

        # accuracy only for non padding tokens
        pred_tokens_non_padding = pred_tokens[gt_caption_ids != self.tokenizer.pad_token_id]
        gt_tokens_non_padding = gt_caption_ids[gt_caption_ids != self.tokenizer.pad_token_id]
        
        token_acc = (pred_tokens_non_padding == gt_tokens_non_padding).float().mean()
        return token_acc
        
    def loss_caption(self, logits: Tensor, target: Tensor, weights=None) -> Tensor:
        loss_config = {'reduction': 'none', 'ignore_index': 0}
        
        loss_per_word = nnf.cross_entropy(
            # nobj, ntoken, vocab -> nobj*token, vocab
            logits.reshape(-1, self.nvocabs),
            # nobj*ntoken
            target.reshape(-1), 
            **loss_config
        )
        # nobj, ntokens
        loss_per_word = loss_per_word.reshape(target.shape)
        if weights is not None:
            # multiply by weight if given
            loss_per_word = loss_per_word * weights.unsqueeze(1)
        # multiply by weight if given
        final_loss = torch.sum(loss_per_word * (target != 0).float()) / torch.sum(
            torch.sum(target != 0).float() + 1e-6
        )
        return final_loss
    
    
    def forward_evaluation(self, instseg_output, assignment, caption_gt, target):
        extra_output = {}
        # prefix tokens
        # final queries is bsize, nqueries, 128
        queries = self.get_queries(instseg_output, assignment, target)
        num_prefix_tokens = 1
        bsize, nqueries, emb_dim = queries.shape
        caption_model_output = OrderedDict()

        attn_masks = [] # to be consistent with training, not used!

        # inference for each sample separately?
        for batch_id in range(bsize):
            inputs, context_feats = [], None
            ############## prepare queries ################
            # get queries for this sample, reshape to 
            sample_queries = queries[batch_id]
            # project to llm embedding dim
            if not self.dont_project_queries:
                sample_queries = self.query_projector(sample_queries)
            sample_queries = sample_queries.unsqueeze(1)
            inputs.append(sample_queries)
            ############# prepare context ################
            # during eval, need feats for all queries
            assignment_all = [(list(range(nqueries)), list(range(nqueries))) for _ in range(bsize)]
            context_feats, inputs, attn_masks, extra_output, num_prefix_tokens = self.prepare_context(instseg_output, target, assignment_all, inputs, attn_masks, num_prefix_tokens, extra_output, 'val')
            ####### use other caption model hidden states#######
            if self.use_other_caption_feats and 'captioner_output' in instseg_output:
                # no need to match the states here, they are for the same set of queries
                # NOTE: here the hidden states are for multiple samples (in training, its for a list of objects across all samples that had captions)
                # nqueries, seqlen, dim
                other_hidden_states = instseg_output['captioner_output']['val_hidden_states'][batch_id][self.use_hidden_state_ndx]
                if context_feats is not None:
                    # join along seqlen dim
                    context_feats = torch.cat([context_feats, other_hidden_states], dim=1)
                else:
                    context_feats = other_hidden_states
            ############ prepare inputs ############
            inputs_embeds = torch.cat(inputs, dim=1)
            
            # nqueries, 1, query_dim
            scene_cap_output = self.transformer.generate(
                # [nqueries, nprefix, querydim]
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=context_feats, #nqueries, contextlength, embdim
                return_dict_in_generate=True,
                output_scores=True, # get beam indices to pick the correct hidden states
                # NOTE: different output_hidden_states, this is for beam search inference
                output_hidden_states=self.output_beam_hidden_states,
                **self.caption_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # update scene output to batch output, append everything in lists
            for key, tensor in scene_cap_output.items():
                caption_model_output[key] = caption_model_output.get(key, []) + [tensor]

        # concat the sequence ids across all samples in the batch
        caption_model_output['sequences'] = torch.cat(caption_model_output['sequences'], dim=0)

        hidden_state_output = None
        num_beams = self.caption_config['num_beams']
        if self.output_beam_hidden_states:
            hidden_state_output = []
            # during inference, this is called decoder_xx, not just hidden_states!
            for batchid, batch_hidden in enumerate(caption_model_output['decoder_hidden_states']):
                # original output was: list for each generation step -> list for each layer -> nqueries, 1, dim
                # store for this sample, for each layer
                sample_hiddens = []
                # for each layer, join along the generation steps 
                for layer_ndx in range(len(batch_hidden[0])):
                    # outputs for all steps in this layer
                    # nqueries*nbeams, nsteps, dim
                    outputs = torch.cat([batch_hidden[step][layer_ndx] for step in range(len(batch_hidden))], dim=1)

                    num_steps = outputs.shape[1]

                    # nqueries, num_beams, num_steps, emb_dim
                    outputs = outputs.reshape(nqueries, num_beams, num_steps, emb_dim)

                    # make it nqueries, numsteps, numbeams, emb_dim -> then index along numbeams
                    outputs = outputs.permute(0, 2, 1, 3)

                    # for query i, for step j, which beam to pick (dim 2) of the hidden states
                    # keep for the same number of steps
                    # nqueries, num_steps
                    beam_ids = caption_model_output['beam_indices'][batchid][:, :num_steps] 

                    # convert beam ids to index within each query -> subtract 0*numbeams from row1, 1*numbeams from row2, etc
                    # 0 to num_beams now
                    beam_ids = beam_ids - torch.arange(nqueries).unsqueeze(1).to(beam_ids.device) * num_beams 
                    # any -1s in the remaining beam ids -> are for incomplete sequences since not all of them have the same length
                    # if any beam id is <0 now, set it to 0 
                    beam_ids[beam_ids < 0] = 0
                    # select the correct hidden states according to beam ids
                    # nqueries, nsteps, dim
                    try:
                        outputs = outputs[torch.arange(nqueries).unsqueeze(1), torch.arange(num_steps), beam_ids]
                    except:
                        # zero features
                        outputs = torch.zeros(nqueries, num_steps, emb_dim).to(outputs.device)
                
                    sample_hiddens.append(outputs)
                hidden_state_output.append(sample_hiddens)

            # NOTE: save into a different name to not clash with train hidden states
            extra_output['val_hidden_states'] = hidden_state_output

        # get text captions, decode everything together
        captions = self.tokenizer.batch_decode(
            # nquery, maxlength
            # no need to remove prefix tokens-1 from the beginning, outputs always start after the inputs
            caption_model_output['sequences'].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # store object captions as text in eval format
        # list [captions for each sample -> [caption for each query]]
        caption_output = [
            [
                captions[batch_id * nqueries + prop_id] \
                    for prop_id in range(nqueries)
            ] \
            for batch_id in range(bsize)
        ]

        return caption_output, extra_output
    
