
try:
    import MinkowskiEngine as ME
except ModuleNotFoundError:
    print('Skipping import of MinkowskiEngine')
import numpy as np
import torch
from random import random

class VoxelizeCollate:
    def __init__(
            self,
            ignore_label=255,
            voxel_size=1,
            mode="test",
            small_crops=False,
            very_small_crops=False,
            batch_instance=False,
            probing=False,
            task="instance_segmentation",
            ignore_class_threshold=100,
            filter_out_classes=[],
            label_offset=0,
            num_queries=None,
            num_classes=None,
            segment_strategy="overlap_thresh",
            segment_overlap_thresh=0.9,
            gen_captions=None,
            gen_part_captions=None
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "task not known"
        self.task = task
        self.gen_captions = gen_captions
        self.gen_part_captions = gen_part_captions
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        print('Creating VoxelizeCollate with mode:', mode)
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold
        # total number of semantic classes
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.segment_strategy = segment_strategy
        self.segment_overlap_thresh = segment_overlap_thresh

    def __call__(self, batch):
        if ("train" in self.mode) and (self.small_crops or self.very_small_crops):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode,
                        task=self.task, ignore_class_threshold=self.ignore_class_threshold,
                        filter_out_classes=self.filter_out_classes, label_offset=self.label_offset,
                        num_queries=self.num_queries, num_classes=self.num_classes,
                        segment_overlap_thresh=self.segment_overlap_thresh, segment_strategy=self.segment_strategy,
                        gen_captions=self.gen_captions, gen_part_captions=self.gen_part_captions)


class VoxelizeCollateMerge:
    def __init__(
            self,
            ignore_label=255,
            voxel_size=1,
            mode="test",
            scenes=2,
            small_crops=False,
            very_small_crops=False,
            batch_instance=False,
            make_one_pc_noise=False,
            place_nearby=False,
            place_far=False,
            proba=1,
            probing=False,
            task="instance_segmentation"
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "task not known"
        self.task = task
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba
        self.probing = probing

    def __call__(self, batch):
        if (
                ("train" in self.mode)
                and (not self.make_one_pc_noise)
                and (self.proba > random())
        ):
            if self.small_crops or self.very_small_crops:
                batch = make_crops(batch)
            if self.very_small_crops:
                batch = make_crops(batch)
            if self.batch_instance:
                batch = batch_instances(batch)
            new_batch = []
            for i in range(0, len(batch), self.scenes):
                batch_coordinates = []
                batch_features = []
                batch_labels = []

                batch_filenames = ""
                batch_raw_color = []
                batch_raw_normals = []

                offset_instance_id = 0
                offset_segment_id = 0

                for j in range(min(len(batch[i:]), self.scenes)):
                    batch_coordinates.append(batch[i + j][0])
                    batch_features.append(batch[i + j][1])

                    if j == 0:
                        batch_filenames = batch[i + j][3]
                    else:
                        batch_filenames = batch_filenames + f"+{batch[i + j][3]}"

                    batch_raw_color.append(batch[i + j][4])
                    batch_raw_normals.append(batch[i + j][5])

                    # make instance ids and segment ids unique
                    # take care that -1 instances stay at -1
                    batch_labels.append(batch[i + j][2] + [0, offset_instance_id, offset_segment_id])
                    batch_labels[-1][batch[i + j][2][:, 1] == -1, 1] = -1

                    max_instance_id, max_segment_id = batch[i + j][2].max(axis=0)[1:]
                    offset_segment_id = offset_segment_id + max_segment_id + 1
                    offset_instance_id = offset_instance_id + max_instance_id + 1

                if (len(batch_coordinates) == 2) and self.place_nearby:
                    border = batch_coordinates[0][:, 0].max()
                    border -= batch_coordinates[1][:, 0].min()
                    batch_coordinates[1][:, 0] += border
                elif (len(batch_coordinates) == 2) and self.place_far:
                    batch_coordinates[1] += (
                            np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                    )
                new_batch.append(
                    (
                        np.vstack(batch_coordinates),
                        np.vstack(batch_features),
                        np.concatenate(batch_labels),
                        batch_filenames,
                        np.vstack(batch_raw_color),
                        np.vstack(batch_raw_normals)
                    )
                )
            # TODO WHAT ABOUT POINT2SEGMENT AND SO ON ...
            batch = new_batch
        elif ("train" in self.mode) and self.make_one_pc_noise:
            new_batch = []
            for i in range(0, len(batch), 2):
                if (i + 1) < len(batch):
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    batch[i][2],
                                    np.full_like(batch[i + 1][2], self.ignore_label),
                                )
                            ),
                        ]
                    )
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    np.full_like(batch[i][2], self.ignore_label),
                                    batch[i + 1][2],
                                )
                            ),
                        ]
                    )
                else:
                    new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
            batch = new_batch
        # return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode)
        return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode, task=self.task)


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch

def get_sem_inst_mappings(num_sem_classes, ignore_classes):
    # create mappings between instance semantic ID (0 to instclasses-1) and original semantic ID (0 to semclasses-1)
    # 0 to semclasses-1 -> 0 to instclasses-1
    semid_to_instsemid = {}
    # 0 to instclasses-1 -> 0 to semclasses-1
    instsemid_to_semid = {}

    for semid in range(num_sem_classes):
        # ignore classes are 1-indexed
        if semid in ignore_classes:
            continue
        # new id = number of classes seen till now
        semid_to_instsemid[semid] = len(semid_to_instsemid)
        # reverse mapping
        instsemid_to_semid[len(instsemid_to_semid)] = semid

    return semid_to_instsemid, instsemid_to_semid


def voxelize(batch, ignore_label, voxel_size, probing, mode, task,
             ignore_class_threshold, filter_out_classes, label_offset, num_queries, num_classes,
             segment_strategy, segment_overlap_thresh, gen_captions, gen_part_captions):
    # original = lists of properties per sample
    (coordinates, features, labels, original_labels, inverse_maps, unique_maps, original_colors, original_normals,
     original_coordinates, idx, cap_gt) = (
        [], [], [], [], [], [], [], [], [], [], []
    )
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []
    
    for sample in batch:
        # sample = coordinates, features, labels, self.data[idx]['scene'], 
        #           raw_color, raw_normals, raw_coordinates, idx
        full_res_coords.append(sample[0])
        # sem+inst+segid
        original_labels.append(sample[2])
        original_colors.append(sample[4])
        original_normals.append(sample[5])
        original_coordinates.append(sample[6])
        # index in the dataset
        idx.append(sample[7])
        # 8th / last element is caption GT, if it exists
        cap_gt.append(sample[8])

        # get coordinates of voxels
        # same shape as sample[0]
        # integers, can be negative
        coords = np.floor(sample[0] / voxel_size)
        # vox coords - not yet unique
        voxelization_dict.update({"coordinates": torch.from_numpy(coords).to("cpu").contiguous(), "features": sample[1]})

        # quantize the coordinates
        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        # return_index: indices of the quantized coordinates = true 
        # return_inverse: indices that can recover the discretized original coordinates = true
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        # usage:
        # unique_map: go from discretize orig coords to final voxel coords (remove duplicates)
        # unique_coords = discrete_coords[unique_map] -> unique_map is <= coords
        # inverse_map: go from final voxel coords to discretized orig coords (have duplicates)
        # unique_coords[inverse_map] == discrete_coords -> inverse map shape is same as orig coords
        inverse_maps.append(inverse_map)
        unique_maps.append(unique_map)

        # final voxel coordinates
        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        # pick the corresponding features
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        # if it has labels, pick the corresponding labels
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long()) 

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        # Create input arguments for a sparse tensor
        # Convert a set of coordinates and features into the batch coordinates and batch features.
        input_dict["labels"] = labels
        # coordinates has extra starting dim to indicate the sample index
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict) 
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

    if probing:
        # quick run just for voxelization?
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, ),
            labels,
        )

    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, 0], return_index=True, return_inverse=True)
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, -1], return_index=True, return_inverse=True)
                # make the segment indices unique and 0-N
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                # go from new segment indices to old ones?
                input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])

    # labels given -> train or val set
    if "labels" in input_dict:
        list_labels = input_dict["labels"]

        target = []
        target_full = []

        # N,
        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append({
                    'labels': label_ids,
                    'masks': list_labels[batch_id] == label_ids.unsqueeze(1)
                })
        else:
            # N,3: seg,inst,segment
            if mode == "test":
                # NOTE: test and train data prep is different!
                for i in range(len(input_dict["labels"])):
                    target.append({
                        "point2segment": input_dict["labels"][i][:, 0]
                    })
                    target_full.append({
                        "point2segment": torch.from_numpy(original_labels[i][:, 0]).long()
                    })
            else:
                # voxel instance masks
                # 'labels', 'masks', 'segment_mask', 'point2segment'
                # labels -> sem labels for each object
                # masks: mask for each object
                # segment_mask: segment mask for each object, segment IDs 0 to N-1
                # point2segment: point (voxel) to segment mapping for each object = segment ID for each voxel 0-N-1
                target = get_instance_masks(list_labels,
                                            list_segments=input_dict["segment2label"],
                                            task=task,
                                            ignore_class_threshold=ignore_class_threshold,
                                            # filter out classes -> according to the mapped semantic classes
                                            # not the original labels!
                                            filter_out_classes=filter_out_classes,
                                            label_offset=label_offset,
                                            num_classes=num_classes,
                                            segment_strategy=segment_strategy,
                                            segment_overlap_thresh=segment_overlap_thresh
                                            )
                for i in range(len(target)):
                    # add the segment info along with masks
                    # segment ID for each voxel
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]
                if "train" not in mode:
                    # point instance masks, for validation and test
                    target_full = get_instance_masks([torch.from_numpy(l) for l in original_labels],
                                                        task=task,
                                                        ignore_class_threshold=ignore_class_threshold,
                                                        filter_out_classes=filter_out_classes,
                                                        label_offset=label_offset,
                                                        num_classes=num_classes,
                                                        segment_strategy=segment_strategy,
                                                        segment_overlap_thresh=segment_overlap_thresh
                                                        )
                    for i in range(len(target_full)):
                        # add the segment info along with masks
                        target_full[i]["point2segment"] = torch.from_numpy(original_labels[i][:, 2]).long()
    else:
        # no labels given, test set?
        target = []
        target_full = []
        coordinates = []
        features = []

    # prepare caption GT
    cap_gt_batch = {}
    if gen_captions:
        cap_gt_batch['cap_gt_corpus'] = [c['cap_gt_corpus'] for c in cap_gt]
        cap_gt_batch['cap_obj_ids'] = [c['cap_obj_ids'] for c in cap_gt]
        cap_gt_batch['cap_sem_ids'] = torch.cat([torch.LongTensor(c['cap_sem_ids']) for c in cap_gt])
        cap_gt_batch['cap_gt_tokens'] = torch.cat([torch.LongTensor(c['cap_gt_tokens']) for c in cap_gt])
        cap_gt_batch['cap_gt_attn_mask'] = torch.cat([torch.FloatTensor(c['cap_gt_attn_mask']) for c in cap_gt])
    if gen_part_captions:
        cap_gt_batch['part_cap_gt_corpus'] = [c['part_cap_gt_corpus'] for c in cap_gt]
        cap_gt_batch['part_cap_obj_ids'] = [c['part_cap_obj_ids'] for c in cap_gt]
        cap_gt_batch['part_cap_sem_ids'] = torch.cat([torch.LongTensor(c['part_cap_sem_ids']) for c in cap_gt])
        cap_gt_batch['part_cap_gt_tokens'] = torch.cat([torch.LongTensor(c['part_cap_gt_tokens']) for c in cap_gt])
        cap_gt_batch['part_cap_gt_attn_mask'] = torch.cat([torch.FloatTensor(c['part_cap_gt_attn_mask']) for c in cap_gt])

    if "train" not in mode:
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, unique_maps, full_res_coords,
                  target_full, original_colors, original_normals, original_coordinates, idx), target,
            [sample[3] for sample in batch], #scene IDs
            cap_gt_batch
        )
    else:
        # train doesnt have target full, original colors, normals, coordinates, idx
        # changed: need original coordinates for 2d feat!
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, unique_maps, full_res_coords,
                  original_coordinates=original_coordinates), target,
            [sample[3] for sample in batch], #scene IDs
            cap_gt_batch
        )


def get_instance_masks(list_labels, task, list_segments=None, ignore_class_threshold=100,
                       filter_out_classes=[], label_offset=0, num_classes=None, segment_overlap_thresh=None,
                       segment_strategy=None):
    '''
    seg_overlap_thresh: this fraction of the segment should be inside the object
    '''
    # filter_out_classes: 0-indexed semantic ID of classes to ignore
    target = []

    # create mapping from semantic to instance-sem IDs
    # mapping from (0-semantic classes) to (0-num inst classes)
    if num_classes is not None:
        semid_to_instsemid, _ = get_sem_inst_mappings(num_classes, filter_out_classes)

    # each sample
    for batch_id in range(len(list_labels)):
        label_ids = []
        inst_ids = []
        masks = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()
        
        # each object in sample
        for instance_id in instance_ids:
            if instance_id == -1:
                continue
            
            # works for ours -> instance ID = -100
            if instance_id < 0:
                continue

            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            # get the labels for this instance
            tmp = list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id]
            # semantic ID of this instance = first element of the first row
            label_id = tmp[0, 0]

            if label_id in filter_out_classes:  # floor, wall, undefined==255 is not included
                continue

            if 255 in filter_out_classes and label_id.item() == 255 and tmp.shape[0] < ignore_class_threshold:
                continue

            label_ids.append(label_id)
            inst_ids.append(instance_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            if list_segments:
                # store mask of segments that are "in use" by instances
                segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                # voxels in this instance
                inst_mask = list_labels[batch_id][:, 1] == instance_id
                # find the segments that belong to this object
                obj_segids = list_labels[batch_id][inst_mask, 2].unique()

                for segid in obj_segids:
                    seg_mask = list_labels[batch_id][:, 2] == segid

                    keep_segment = False
                    
                    if segment_strategy == 'keep_all':
                        keep_segment = True
                    elif segment_strategy == 'overlap_thresh':
                        # keep only segments that mostly overlap with this object! otherwise we end up 
                        # with some extra segments that could be mostly outside the object -> dont assume that the segments are 
                        # the GT segments used for annotation
                        # voxels in this seg
                        seg_in_inst_frac = (seg_mask & inst_mask).sum().float() / seg_mask.sum().float()
                        if segment_strategy == 'overlap_thresh' and seg_in_inst_frac > segment_overlap_thresh:
                            keep_segment = True
                    elif segment_strategy == 'majority_instance':
                        # get the unique instance ids in this segment
                        seg_inst_ids = list_labels[batch_id][:, 1][seg_mask]
                        # get the instance id that has the most voxels in this segment
                        mode_inst_id = torch.mode(seg_inst_ids).values
                        # segment's mode instance id is the instance id of this object
                        if mode_inst_id == instance_id:
                            keep_segment = True
                    if keep_segment:
                        segment_mask[segid] = True

                segment_masks.append(segment_mask)

        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        inst_ids = torch.stack(inst_ids)
        masks = torch.stack(masks)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = (label_ids == label_id)

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(segment_masks[masking, :].sum(dim=0).bool())

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)

                target.append({
                    'labels': label_ids,
                    'masks': masks,
                    'segment_mask': segment_masks
                })
            else:
                target.append({
                    'labels': label_ids,
                    'masks': masks
                })
        else:
            if num_classes is not None:
                # for scannetpp, use class mapping, dont use label offset!
                l = torch.clone(label_ids)
                # semantic IDs to instance-semantic IDs
                for label in semid_to_instsemid:
                    # compare in the original tensor
                    # new tensor could have changed
                    # TODO: should have +1 here?
                    l[label_ids == label] = semid_to_instsemid[label]
            else:
                l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append({
                    'labels': l,
                    'masks': masks,
                    'segment_mask': segment_masks,
                    'inst_ids': inst_ids
                })
            else:
                target.append({
                    'labels': l,
                    'masks': masks
                })
    return target


def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class NoGpu:
    def __init__(
            self, coordinates, features, original_labels=None, inverse_maps=None, 
            unique_maps=None, full_res_coords=None,
            target_full=None, original_colors=None, original_normals=None, original_coordinates=None,
            idx=None, cap_gt=None
    ):
        """ helper class to prevent gpu loading on lightning """
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.unique_maps = unique_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
        self.cap_gt = cap_gt


class NoGpuMask:
    def __init__(
            self, coordinates, features, original_labels=None, inverse_maps=None, masks=None, labels=None
    ):
        """ helper class to prevent gpu loading on lightning """
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels