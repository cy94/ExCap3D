import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import Optional, Tuple, Union
from random import choice
from copy import deepcopy
from transformers import GPT2Tokenizer, T5Tokenizer

from scannetpp.common.file_io import load_json
import torch
from datasets.random_cuboid import RandomCuboid

try:
    import albumentations as A
    import volumentations as V
except:
    pass
import numpy as np
import scipy
import yaml

from yaml import FullLoader
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
)

logger = logging.getLogger(__name__)


def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
        overfit=False,
        overfit_n_instances=None,
        clip_points=0,
        subset=None,
        keep_instance_classes_file=None,
        no_aug=None,
        list_file=None,
        caption_data_dir=None,
        max_caption_length=None,
        repeat=None,
        semlabel_caption=False,
        exclude_scenes_without_caption=False,
        tokenizer_model_name='gpt2',
        gen_captions=False,
        gen_part_captions=False,
        overfit_instance_ids=None,
        obj_caption_key=None,
        part_caption_key=None,
        caption_no_aug=False,
        max_caption_aug=None,
        extra_feats_dir=None,
        extra_feats_dim=None,
        scannet_use_others_class=False,
    ):
        self.scannet_use_others_class = scannet_use_others_class

        self.extra_feats_dir = extra_feats_dir
        self.extra_feats_dim = extra_feats_dim

        self.max_caption_aug = max_caption_aug
        # key in json to get captions
        self.caption_no_aug = caption_no_aug
        self.obj_caption_key = obj_caption_key
        self.part_caption_key = part_caption_key
        
        self.gen_captions = gen_captions
        self.gen_part_captions = gen_part_captions
        self.exclude_scenes_without_caption = exclude_scenes_without_caption
        self.semlabel_caption = semlabel_caption
        self.max_caption_length = max_caption_length

        self.repeat = repeat
        self.scene_list = read_txt_list(list_file)

        self.keep_instance_classes = None
        if keep_instance_classes_file:
            self.keep_instance_classes = read_txt_list(keep_instance_classes_file)
        
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"

        # keep fewer than all the points to reduce memory usage
        self.clip_points = clip_points

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "stpls3d":
            self.color_map = {
                0: [0, 255, 0],  # Ground
                1: [0, 0, 255],  # Build
                2: [0, 255, 255],  # LowVeg
                3: [255, 255, 0],  # MediumVeg
                4: [255, 0, 255],  # HiVeg
                5: [100, 100, 255],  # Vehicle
                6: [200, 200, 100],  # Truck
                7: [170, 120, 200],  # Aircraft
                8: [255, 0, 0],  # MilitaryVec
                9: [200, 100, 100],  # Bike
                10: [10, 200, 100],  # Motorcycle
                11: [200, 200, 200],  # LightPole
                12: [50, 50, 50],  # StreetSign
                13: [60, 130, 60],  # Clutter
                14: [130, 30, 60],
            }  # Fence
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
        elif self.dataset_name == "s3dis":
            self.color_map = {
                0: [0, 255, 0],  # ceiling
                1: [0, 0, 255],  # floor
                2: [0, 255, 255],  # wall
                3: [255, 255, 0],  # beam
                4: [255, 0, 255],  # column
                5: [100, 100, 255],  # window
                6: [200, 200, 100],  # door
                7: [170, 120, 200],  # table
                8: [255, 0, 0],  # chair
                9: [200, 100, 100],  # sofa
                10: [10, 200, 100],  # bookcase
                11: [200, 200, 200],  # board
                12: [50, 50, 50],  # clutter
            }
        elif self.dataset_name == "scannetpp":
            colors = np.random.randint(0, 255, (200, 3))
            self.color_map = {i: color for (i, color) in enumerate(colors)}
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size,
            crop_length=self.crop_length,
            version1=self.version1,
        )

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        self.overfit = overfit
        self.overfit_n_instances = overfit_n_instances
        self.overfit_instance_ids = overfit_instance_ids
        self.is_overfit = self.overfit_n_instances or self.overfit_instance_ids or self.overfit

        self.no_aug = no_aug

        if overfit:
            # use train dataset for both, only 1st sample, no augmentation
            print('Overfitting, set mode to train')
            self.mode = "train"
            print('Set no aug to true')
            self.no_aug = True
            
        print('No augmentation?:', self.no_aug)

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if self.dataset_name != "s3dis":
                if not (database_path / f"{self.mode}_database.yaml").exists():
                    print(
                        f"generate {database_path}/{self.mode}_database.yaml first"
                    )
                    exit()
                self._data.extend(
                    self._load_yaml(database_path / f"{self.mode}_database.yaml")
                )
            else:
                mode_s3dis = f"Area_{self.area}"
                if self.mode == "train":
                    mode_s3dis = "train_" + mode_s3dis
                if not (
                    database_path / f"{mode_s3dis}_database.yaml"
                ).exists():
                    print(
                        f"generate {database_path}/{mode_s3dis}_database.yaml first"
                    )
                    exit()
                self._data.extend(
                    self._load_yaml(
                        database_path / f"{mode_s3dis}_database.yaml"
                    )
                )

        db_samples = len(self._data)
        # keep only the scenes in the scene list
        # different for scannet and scannetpp (see scene_ids)
        if self.dataset_name == 'scannet':
            self._data = [d for d in self._data if f"scene{d['scene']:04d}_{d['sub_scene']:02d}" in self.scene_list]
        elif self.dataset_name == 'scannetpp':
            self._data = [d for d in self._data if d['scene'] in self.scene_list]

        print(f'Kept {len(self._data)} samples out of {db_samples} based on scene list')

        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )

        if self.overfit:
            # keep 1 sample
            self._data = self._data[:1]
            print('Overfitting, keeping only 1 sample')
            print(self._data[0]['filepath'])
            # train set: repeat it 100 times
            # mode specified was originally train
            if mode == "train":
                print('Repeat train sample 100 times for overfitting')
                self._data = self._data * 100

        if not self.overfit and self.repeat is not None:
            print(f'Repeating dataset {self.repeat} times, original size: {len(self._data)}')
            self._data = self._data * self.repeat
            print(f'New size: {len(self._data)}')

        # caption data: raw captions, corpus=tokenized and prepared for training
        self.caption_data, self.caption_corpus, self.part_caption_corpus = None, None, None

        if self.gen_captions or self.gen_part_captions:
            if 'gpt' in tokenizer_model_name:
                self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif 't5' in tokenizer_model_name: # auto
                self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name)

            caption_data_dir = Path(caption_data_dir)
            # get list of unique scene ids in the dataset
            unique_scene_ids = set(self.scene_ids)
            # scenes which have captions
            scenes_with_captions = [scene_id for scene_id in unique_scene_ids if (caption_data_dir / f'{scene_id}.json').is_file()]

            if self.exclude_scenes_without_caption:
                # adjust self._data again
                self._data = [d for d in self._data if self.scene_data_to_scene_id(d) in scenes_with_captions]
                print(f'Keeping {len(self._data)} scenes with captions')

            self.caption_data = {scene_id: load_json(caption_data_dir / f'{scene_id}.json') for scene_id in scenes_with_captions}

            self.caption_corpus = {}
            self.part_caption_corpus = {}

            # clip captions to required length 
            for scene_id, scene_data in self.caption_data.items():
                for object_id, object_data in scene_data['objects'].items():
                    key = (scene_id, object_id)

                    if self.gen_captions:
                        # the "actual" caption used for training, use this later for evaluation
                        caption = object_data[self.obj_caption_key]

                        if type(caption) == list: # list of captions
                            self.caption_corpus[key] = None # do on the fly
                        else: # single caption
                            self.caption_corpus[key] = self.get_clipped_caption(caption)

                    if self.gen_part_captions:
                        # not all objects have part captions
                        if self.part_caption_key in object_data:
                            # TODO: add caption sampling just like above
                            caption = object_data[self.part_caption_key]
                            if type(caption) == list:
                                caption = caption[0]
                            self.part_caption_corpus[(scene_id, object_id)] = self.get_clipped_caption(caption)

            print(f'Loaded captions for {len(self.caption_data)} scenes')

        self.subset = subset
        if self.subset:
            print('Keep subset of dataset:', self.subset)
            self._data = self._data[:self.subset]

        labels = self._load_yaml(Path(self.data_dir[0]) / 'label_database.yaml')

        if self.scannet_use_others_class:
            print('>>>>>>> Using scannet otherprop and otherstructure classes for training and evaluation')
            # set validation to true for otherprop and otherstructure classes
            for nyuid, label_data in labels.items():
                if label_data['name'] in ['otherprop', 'otherstructure']:
                    label_data['validation'] = True

        # if working only on classes for validation - discard others
        print('Original labels:', len(labels))
        # select only the ones to train on
        # dict: orig label -> {name, validation, color}
        self._labels = self._select_correct_labels(labels, num_labels)
        print('Selected labels:', len(self._labels), [(k, v['name']) for k, v in self._labels.items()])

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(self.data_dir[0]) / "instance_database.yaml"
            )

        # normalize color channels
        if self.dataset_name == "s3dis":
            color_mean_std = color_mean_std.replace(
                "color_mean_std.yaml", f"Area_{self.area}_color_mean_std.yaml"
            )

        color_mean_std_path = Path(self.data_dir[0]) / "color_mean_std.yaml"

        if Path(str(color_mean_std_path)).exists():
            color_mean_std = self._load_yaml(color_mean_std_path)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error(
                "pass mean and std as tuple of tuples, or as an .yaml file"
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color normalization
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data
        # new_data = []
        if self.cache_data:
            new_data = []
            for i in range(len(self._data)):
                self._data[i]["data"] = np.load(
                    self.data[i]["filepath"].replace("../../", "")
                )
                if self.on_crops:
                    if self.eval_inner_core == -1:
                        for block_id, block in enumerate(
                            self.splitPointCloud(self._data[i]["data"])
                        ):
                            if len(block) > 10000:
                                new_data.append(
                                    {
                                        "instance_gt_filepath": self._data[i][
                                            "instance_gt_filepath"
                                        ][block_id]
                                        if len(
                                            self._data[i][
                                                "instance_gt_filepath"
                                            ]
                                        )
                                        > 0
                                        else list(),
                                        "scene": f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                        "raw_filepath": f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                        "data": block,
                                    }
                                )
                            else:
                                assert False
                    else:
                        conds_inner, blocks_outer = self.splitPointCloud(
                            self._data[i]["data"],
                            size=self.crop_length,
                            inner_core=self.eval_inner_core,
                        )

                        for block_id in range(len(conds_inner)):
                            cond_inner = conds_inner[block_id]
                            block_outer = blocks_outer[block_id]

                            if cond_inner.sum() > 10000:
                                new_data.append(
                                    {
                                        "instance_gt_filepath": self._data[i][
                                            "instance_gt_filepath"
                                        ][block_id]
                                        if len(
                                            self._data[i][
                                                "instance_gt_filepath"
                                            ]
                                        )
                                        > 0
                                        else list(),
                                        "scene": f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                        "raw_filepath": f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                        "data": block_outer,
                                        "cond_inner": cond_inner,
                                    }
                                )
                            else:
                                assert False

            if self.on_crops:
                self._data = new_data
                # new_data.append(np.load(self.data[i]["filepath"].replace("../../", "")))
            # self._data = new_data


    def get_clipped_caption(self, caption):
        caption_tokens = self.tokenizer.encode(caption, max_length=self.max_caption_length-1)
        keep_num_tokens = min(len(caption_tokens), self.max_caption_length)
        clipped_caption = self.tokenizer.decode(caption_tokens[:keep_num_tokens])
        return clipped_caption

    def get_clipped_caption_multiple(self, caption_list):
        tokenizer_output = self.tokenizer.batch_encode_plus(
                    caption_list, 
                    max_length=self.max_caption_length-1, 
                    padding='max_length', 
                    truncation='longest_first', 
                    return_tensors='np'
                )
        # decode back to required length
        clipped_captions = self.tokenizer.batch_decode(
            tokenizer_output['input_ids'].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return clipped_captions

    def splitPointCloud(self, cloud, size=50.0, stride=50, inner_core=-1):
        if inner_core == -1:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - size) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks = []
            for (x, y) in cells:
                xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
                ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
                cond = xcond & ycond
                block = cloud[cond, :]
                blocks.append(block)
            return blocks
        else:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks_outer = []
            conds_inner = []
            for (x, y) in cells:
                xcond_outer = (
                    cloud[:, 0] <= x + inner_core / 2.0 + size / 2
                ) & (cloud[:, 0] >= x + inner_core / 2.0 - size / 2)
                ycond_outer = (
                    cloud[:, 1] <= y + inner_core / 2.0 + size / 2
                ) & (cloud[:, 1] >= y + inner_core / 2.0 - size / 2)

                cond_outer = xcond_outer & ycond_outer
                block_outer = cloud[cond_outer, :]

                xcond_inner = (block_outer[:, 0] <= x + inner_core) & (
                    block_outer[:, 0] >= x
                )
                ycond_inner = (block_outer[:, 1] <= y + inner_core) & (
                    block_outer[:, 1] >= y
                )

                cond_inner = xcond_inner & ycond_inner

                conds_inner.append(cond_inner)
                blocks_outer.append(block_outer)
            return conds_inner, blocks_outer

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return 5 * len(self.data)
        else:
            return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int, return_gt_data=False):
        '''
        return_gt_data: return only the gtdata used for evaluation (1000sem+inst+1)
        '''
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        scene_id = self.scene_ids[idx]

        if self.cache_data:
            points = self.data[idx]["data"]
        else:
            assert not self.on_crops, "you need caching if on crops"
            points = np.load(self.data[idx]["filepath"].replace("../../", ""))

        if "train" in self.mode and self.dataset_name in ["s3dis", "stpls3d"]:
            inds = self.random_cuboid(points)
            points = points[inds]

        extra_feats = None
        if self.extra_feats_dir is not None:
            feats_path = Path(self.extra_feats_dir) / f'{scene_id}.pth'
            if feats_path.exists():
                extra_feats = torch.load(feats_path, map_location='cpu')
                # if tensor, make cpu and numpy
                if isinstance(extra_feats, torch.Tensor):
                    extra_feats = extra_feats.cpu().numpy()
            else:
                extra_feats = np.zeros((len(points), self.extra_feats_dim))

        # clip points before doing anything else
        # only for train
        if self.clip_points > 0 and "train" in self.mode:
            if len(points) > self.clip_points:
                # randomly sample and keep fewer points
                ndx = np.random.choice(len(points), self.clip_points, replace=False)
                points = points[ndx]

                if extra_feats is not None: #NOTE: anywhere points are clipped, need to clip extra_feats as well
                    extra_feats = extra_feats[ndx]

        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        if self.no_aug:
            # normalize coordinates because its not done below
            # TODO: is this done only for train?
            coordinates -= coordinates.mean(0)

        # volume and image augmentations for train
        # + other random changes
        if not self.no_aug and not self.overfit and "train" in self.mode or self.is_tta:
            if self.cropping: # default not used
                new_idx = self.random_cuboid(coordinates)
                #     labels[:, 1],
                #     self._remap_from_zero(labels[:, 0].copy()),
                # )

                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]

                if extra_feats is not None:
                    extra_feats = extra_feats[new_idx]

            # normalize coordinates
            coordinates -= coordinates.mean(0)

            # random shift of the point cloud
            try:
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0))
                    / 2
                )
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            if self.instance_oversampling > 0.0: # default not used
                (
                    coordinates,
                    color,
                    normals,
                    labels,
                ) = self.augment_individual_instance(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.instance_oversampling,
                )

            if self.flip_in_center: #default not used
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                # flip coordinate wrt max
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            # random elastic
            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            # volume augmentation, noop if not set
            aug = self.volume_augmentations(
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            # convert point colors to "image"
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            # and apply image augmentations
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

            if self.point_per_cut != 0: # default not used
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )
            # add more random points, resample
            if (self.resample_points > 0) or (self.noise_rate > 0): # default not used
                coordinates, color, normals, labels = random_around_points(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if self.add_unlabeled_pc: #default not used
                if random() < 0.8:
                    new_points = np.load(
                        self.other_database[
                            np.random.randint(0, len(self.other_database) - 1)
                        ]["filepath"]
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        new_points[:, :3],
                        new_points[:, 3:6],
                        new_points[:, 6:9],
                        new_points[:, 9:],
                    )
                    unlabeled_coords -= unlabeled_coords.mean(0)
                    unlabeled_coords += (
                        np.random.uniform(
                            unlabeled_coords.min(0), unlabeled_coords.max(0)
                        )
                        / 2
                    )

                    aug = self.volume_augmentations(
                        points=unlabeled_coords,
                        normals=unlabeled_normals,
                        features=unlabeled_color,
                        labels=unlabeled_labels,
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        aug["points"],
                        aug["features"],
                        aug["normals"],
                        aug["labels"],
                    )
                    pseudo_image = unlabeled_color.astype(np.uint8)[
                        np.newaxis, :, :
                    ]
                    unlabeled_color = np.squeeze(
                        self.image_augmentations(image=pseudo_image)["image"]
                    )

                    coordinates = np.concatenate(
                        (coordinates, unlabeled_coords)
                    )
                    color = np.concatenate((color, unlabeled_color))
                    normals = np.concatenate((normals, unlabeled_normals))
                    labels = np.concatenate(
                        (
                            labels,
                            np.full_like(unlabeled_labels, self.ignore_label),
                        )
                    )

            # reset colors to fixed value
            if random() < self.color_drop: # default not used
                color[:] = 255

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        # keep the orig sem+inst labels
        labels_orig = labels.copy()

        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            # make instance invalid wherever sem is invalid
            sem_invalid = labels[:, 0] == self.ignore_label
            labels[sem_invalid, 1] = self.ignore_label
            # set original labels as well, sem->0, inst->-1
            labels_orig[sem_invalid, 0] = 0
            labels_orig[sem_invalid, 1] = -1

            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color # default feature is color
        if self.add_normals: # default false
            features = np.hstack((features, normals))
        if self.add_raw_coordinates: # default true
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
         # just add the feats here, figure out during training whether to use them or not
        if extra_feats is not None:
            features = np.hstack((features, extra_feats))

        # scannet bug scenes?
        if self.data[idx]["raw_filepath"].split("/")[-2] in [
            "scene0636_00",
            "scene0154_00",
        ]:
            return self.__getitem__(0)

        if self.overfit_n_instances is not None or self.overfit_instance_ids is not None:
            # labels = sem, inst, segment
            # keep the first n instances which has valid semantic labels
            valid_sem = labels[:, 0] != self.ignore_label
            # find the instance IDs within valid sem, there might still be ignore label here if some sem labels are not instances?
            unique_instance_ids = np.unique(labels[valid_sem, 1])
            # discard ignore label
            unique_instance_ids = unique_instance_ids[unique_instance_ids != self.ignore_label]

            # if generating captions, keep only instances that have captions
            if self.caption_data and scene_id in self.caption_data:
                objs_with_cap = list(map(int, self.caption_data[scene_id]['objects'].keys()))
                # inst that are in the scene and have cap
                inst_with_cap = [i for i in unique_instance_ids if i in objs_with_cap] 
                unique_instance_ids = np.array(inst_with_cap, dtype=int)

            if self.overfit_n_instances:
                # keep n instances to overfit
                keep_instance_ids = unique_instance_ids[:self.overfit_n_instances]
            elif self.overfit_instance_ids:
                # keep only the specified instances
                keep_instance_ids = np.array(self.overfit_instance_ids, dtype=int)

            # fill self.ignore_label in instance labels which are not in keep_instance_ids
            valid_instance = np.isin(labels[:, 1], keep_instance_ids)
            discard_instance = ~valid_instance
            # labels -> set to ignore label
            labels[discard_instance, 1] = self.ignore_label
            # set semantic label to ignore for these points
            labels[discard_instance, 0] = self.ignore_label

            # orig labels -> set something else! sem=0, inst=-1
            # modify the original labels in the same way, to be used for evaluation
            # set sem to 0
            # TODO: is this correct?
            labels_orig[discard_instance, 0] = 0
            # inst ID starts from 1 -> setting inst ID to -1 makes the instance invalid in the GT (after+1 later)
            labels_orig[discard_instance, 1] = -1

        # empty values for scenes that dont have captions, so that everything can 
        # be properly concatenated later
        cap_data_final = {
            'cap_gt_corpus': [],
            'cap_obj_ids': [],
            'cap_sem_ids': [],
            'cap_gt_tokens': np.empty((0, self.max_caption_length), dtype=np.int64),
            'cap_gt_attn_mask': np.empty((0, self.max_caption_length), dtype=np.int64),
            'part_cap_gt_corpus': [],
            'part_cap_obj_ids': [],
            'part_cap_sem_ids': [],
            'part_cap_gt_tokens': np.empty((0, self.max_caption_length), dtype=np.int64),
            'part_cap_gt_attn_mask': np.empty((0, self.max_caption_length), dtype=np.int64)
        }

        if self.gen_captions or self.gen_part_captions:
            # load the captions for these instances
            unique_instance_ids = np.unique(labels[:, 1])
            unique_instance_ids = unique_instance_ids[unique_instance_ids != self.ignore_label]

            # use for training
            gt_captions, gt_corpus_captions, objs_with_cap, obj_sem_ids = [], [], [], []
            part_gt_captions, part_gt_corpus_captions, part_objs_with_cap, part_obj_sem_ids = [], [], [], []

            all_obj_cap_data = self.caption_data[scene_id]['objects']

            for inst_id in unique_instance_ids:
                # semantic ID of the object
                sem_id = labels[labels[:, 1] == inst_id, 0][0]

                # get gt_caption (for training) and corpus_caption (for eval)

                if self.gen_captions:
                    if self.semlabel_caption:
                        # use the sem label as the caption
                        sem_id_orig = labels_orig[labels[:, 1] == inst_id, 0][0]
                        # sem_id_orig = self._remap_model_output([sem_id])
                        sem_label = self.label_info[sem_id_orig]['name']
                        # raw caption used for training
                        gt_caption = sem_label
                        # caption used for evaluation
                        corpus_caption = sem_label 
                    # check if object has a caption
                    elif str(inst_id) in all_obj_cap_data:
                        corpus_key = (scene_id, str(inst_id))

                        # caption used for training, gets clipped now (different from the one in corpus-how?)
                        caption = all_obj_cap_data[str(inst_id)][self.obj_caption_key] 

                        if type(caption) == list:
                            if self.max_caption_aug: # keep only a few captions for training/evaluation
                                n_keep = min(len(caption), self.max_caption_aug)
                                caption = caption[:n_keep] 
                            # sample caption for training, first one for val
                            ndx = np.random.randint(len(caption)) if (self.mode == 'train' and not self.is_overfit and not self.caption_no_aug) else 0
                            gt_caption = caption[ndx] # used for training + getting val loss/etc
                            # eval against all GT captions
                            corpus_caption = self.get_clipped_caption_multiple(caption) 
                        else:
                            gt_caption = caption # only 1 cap
                            # always eval against a list of captions, more generic
                            corpus_caption = [self.caption_corpus[corpus_key]] 
                    else:
                        # no obj cap -> no part cap
                        continue

                    # accumulate obj captions
                    gt_corpus_captions.append(corpus_caption)
                    gt_captions.append(gt_caption)
                    objs_with_cap.append(inst_id)
                    obj_sem_ids.append(int(sem_id))

                # get part caption and accumulate
                if str(inst_id) in all_obj_cap_data and self.gen_part_captions and self.part_caption_key in all_obj_cap_data[str(inst_id)]:
                    part_caption = all_obj_cap_data[str(inst_id)][self.part_caption_key]
                    # TODO: add sampling during train and 1st cap during val selection for parts
                    if type(part_caption) == list:
                        part_caption = part_caption[0]
                    part_corpus_caption = self.part_caption_corpus[(scene_id, str(inst_id))]

                    part_gt_corpus_captions.append(part_corpus_caption)
                    part_gt_captions.append(part_caption)
                    part_objs_with_cap.append(inst_id)
                    part_obj_sem_ids.append(int(sem_id))

            # TODO: not using gt_captions, why is this requried, just use corpus caption
            if gt_captions: # obj captions found
                # get the caption token ids and mask for training
                tokenizer_output = self.tokenizer.batch_encode_plus(
                    gt_captions, 
                    max_length=self.max_caption_length, 
                    padding='max_length', 
                    truncation='longest_first', 
                    return_tensors='np'
                )

                cap_data_final.update({
                    'cap_gt_corpus': gt_corpus_captions,
                    'cap_obj_ids': objs_with_cap,
                    'cap_sem_ids': obj_sem_ids,
                    'cap_gt_tokens': tokenizer_output['input_ids'], # nobj, max_len
                    'cap_gt_attn_mask': tokenizer_output['attention_mask'] # nobj, max_len
                    
                })

            if part_gt_corpus_captions: # part captions found
                part_tokenizer_output = self.tokenizer.batch_encode_plus(
                    part_gt_captions, 
                    max_length=self.max_caption_length, 
                    padding='max_length', 
                    truncation='longest_first', 
                    return_tensors='np'
                )
                cap_data_final.update({
                    'part_cap_gt_corpus': part_gt_corpus_captions,
                    'part_cap_obj_ids': part_objs_with_cap,
                    'part_cap_sem_ids': part_obj_sem_ids,
                    'part_cap_gt_tokens': part_tokenizer_output['input_ids'], # nobj, max_len
                    'part_cap_gt_attn_mask': part_tokenizer_output['attention_mask'] # nobj, max_len
                })

        if return_gt_data:
            sem_label, inst_label = labels_orig[:, 0], labels_orig[:, 1]
            return sem_label * 1000 + inst_label + 1

        if self.dataset_name == "s3dis":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["area"] + "_" + self.data[idx]["scene"],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )
        if self.dataset_name == "stpls3d":
            if labels.shape[1] != 1:  # only segments --> test set!
                if np.unique(labels[:, -2]).shape[0] < 2:
                    print("NO INSTANCES")
                    return self.__getitem__(0)
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["scene"],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )
        if self.dataset_name == 'scannetpp':
            # data[idx][scene] is the scene id
            # coords, colors+coords, sem+inst+segid, sceneid, RGB, normals, sample idx (could be different from the request sample ndx), caption data
            return coordinates, features, labels, self.data[idx]['scene'], \
                   raw_color, raw_normals, raw_coordinates, idx, cap_data_final
        else:
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-2],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
                cap_data_final
            )

    def scene_data_to_scene_id(self, d):
        if self.dataset_name == 'scannet':
            return f"scene{d['scene']:04d}_{d['sub_scene']:02d}" 
        elif self.dataset_name == 'scannetpp':
            return d["scene"]

    @property
    def scene_ids(self):
        scene_ids = [self.scene_data_to_scene_id(d) for d in self.data]
        return scene_ids

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader=FullLoader)
            # file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        print('Num labels for validation:', number_of_validation_labels)

        if num_labels == number_of_all_labels:
            print('Num labels = all labels, keep everything')
            return labels
        elif num_labels == number_of_validation_labels:
            print('Num labels = validation labels, keep only validation labels')
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            # for scannetpp, additional option to keep only some classes 
            # out of everything that is in the data
            # according to self.keep_instance_classes
            if self.dataset_name == 'scannetpp':
                print('Scannetpp: keep only specified classes')
                # keep only the classes in self.keep_instance_classes
                valid_labels = dict()
                for (
                    k,
                    v,
                ) in labels.items():
                    if v["name"] in self.keep_instance_classes:
                        valid_labels.update({k: v})
                print(f'Kept: {len(valid_labels)} classes')
                return valid_labels
            else:
                msg = f"""not available number labels, select from:
                {number_of_validation_labels}, {number_of_all_labels}"""
                raise ValueError(msg)

    def _remap_from_zero(self, labels):
        # labels not in label_info are set to ignore_label
        # label info = all the possible labels (including the ones we dont train on)
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label

        # remap to the range from 0-(N-1)
        # first create a copy
        labels_out = labels.copy()

        # labelinfo is a dict! is the order guaranteed? yes, but better to use a list / dict with 0-N-1 keys?
        for new_label, orig_label in enumerate(self.label_info.keys()):
            mask = labels == orig_label
            labels_out[mask] = new_label
        # eg 4,5,255 -> 1,2,255
        return labels_out

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()

        # orig_label = 1,2,255
        # new_label = 4,5,255
        for orig_label, new_label in enumerate(self.label_info.keys()):
            output_remapped[output == orig_label] = new_label

        return output_remapped

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[
                        labels[:, 1] == choice(np.unique(labels[:, 1]))
                    ]
                )
            else:
                center = np.array(
                    [uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)]
                )
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = (
                instance[:, :3] - instance[:, :3].mean(axis=0) + center
            )
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(1, 0, 0)
                    ),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(0, 1, 0)
                    ),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(
            noise, blurx, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blury, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blurz, mode="constant", cval=0
        )

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])[
            "points"
        ]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])[
            "points"
        ]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates,
    color,
    normals,
    labels,
    rate=0.2,
    noise_rate=0,
    ignore_label=255,
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int(
        (max(max_boundary) - min(min_boundary)) / noise_rate
    )

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(
                    min_boundary[0], max_boundary[0], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[1], max_boundary[1], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[2], max_boundary[2], noisy_coordinates
                ),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full(
        (noisy_coordinates.shape[0], labels.shape[1]), ignore_label
    )

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels
