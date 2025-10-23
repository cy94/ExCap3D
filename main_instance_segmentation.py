import logging
import os
from hashlib import md5
from uuid import uuid4
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything
import random, string
import pytorch_lightning

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines


def get_parameters(cfg: DictConfig):
    # setup scannetpp dataset
    # NOTE
    # num_labels: used in semseg dataset to check if using all or selected labels
    #               and trainer
    # num_targets: goes to the mask3d model as "num_classes", predict the semantic class of instance (C+1 from paper)
    # filter_out_classes: used in voxelizecollate
    # label_offset: subtracted and clamped in voxelizecollate for scannet. for ours, use mapping, dont subtract
    if cfg.data.get('semantic_classes_file', None) and cfg.data.get('instance_classes_file', None):
        # print('*************************************')
        # print('Setting up ScannetPP dataset')
        semantic_classes = read_txt_list(cfg.data.semantic_classes_file)
        instance_classes = read_txt_list(cfg.data.instance_classes_file)
        # print('Num semantic classes:', len(semantic_classes))
        # print('Num instance classes:', len(instance_classes))

        # ignore_sem_classes = [i for i, c in enumerate(semantic_classes) if c not in instance_classes]
        # sem classes to ignore for instances AFTER MAPPING (not in the original labels)
        # ours: when mapping to only instance classes, dont filter out anything
        ignore_sem_classes = []
        # print('****** filter_out_classes:', ignore_sem_classes)

        # set filter_out_classes, label_offset, indoor/num_labels, general.num_targets
        # for train and val
        cfg.data.train_dataset.filter_out_classes = ignore_sem_classes
        cfg.data.validation_dataset.filter_out_classes = ignore_sem_classes
        
        num_targets = len(instance_classes) + 1
        # print('****** num_targets:', num_targets) 
        cfg.general.num_targets = num_targets

        # print('****** num_labels:', len(instance_classes))
        # need to change this everywhere? no, changes automatically in the hydra cfg
        cfg.data.num_labels = len(instance_classes)

        # print('****** label_offset:', 0)
        cfg.data.train_dataset.label_offset = 0
        cfg.data.validation_dataset.label_offset = 0
        # print('*************************************')

    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # generate a random str for experiment name
    random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # create unique id for experiments that are run locally
    if cfg.general.get('use_existing_expt', False):
        # this sets the ckpt dir as well
        # convert to string, hydra doesnt do this??
        cfg.general.experiment_name = str(cfg.general.use_existing_expt)
        if cfg.general.get('resume', False):
            print('Resuming existing run:', cfg.general.use_existing_expt)
        else:
            print('Using existing experiment name:', cfg.general.use_existing_expt)

    else:
        slurm_id = os.getenv('SLURM_JOB_ID')
        slurm_name = os.getenv('SLURM_JOB_NAME')
        
        if slurm_name == 'interactive':
            # add a random string
            run_name = f'{slurm_id}_interactive_{random_str}'
        else:
            if slurm_id:
                # on slurm job, use the id
                run_name = f'{slurm_id}'
            else:
                # local pc, no slurm name or id
                run_name = random_str

        # set experiment name
        cfg.general.experiment_name = run_name
        # add suffix if provided
        if cfg.general.experiment_name_suffix:
            cfg.general.experiment_name += f"_{cfg.general.experiment_name_suffix}"
        print('Created new experiment name:', cfg.general.experiment_name)

    if cfg.data.overfit:
        print('Train overfit:', cfg.data.train_dataset.overfit)
        print('Validation overfit:', cfg.data.validation_dataset.overfit)
        print('Train overfit instances:', cfg.data.train_dataset.overfit_n_instances)
        print('Validation overfit instances:', cfg.data.validation_dataset.overfit_n_instances)

    if not os.path.exists(cfg.general.save_dir) and not cfg.general.no_output: # no output -> dont create dir, purely debugging!
        os.makedirs(cfg.general.save_dir)

    if cfg.general.get('resume', False):
        print('Resuming from last epoch checkpoint for run:', cfg.general.experiment_name)
        cfg['trainer']['resume_from_checkpoint'] = cfg.general.checkpoint

    if not cfg.general.no_log:
        for log in cfg.logging:
            print('Log:', OmegaConf.to_container(log, resolve=True))
            # if resuming, dont use wandb run notes, keep existing
            if log._target_ == 'pytorch_lightning.loggers.WandbLogger' and cfg.general.resume:
                log.notes = None
            loggers.append(hydra.utils.instantiate(log))
            loggers[-1].log_hyperparams(
                flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        print('Load backbone from checkpoint')
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        # NOTE: when resuming, weights get loaded here and again through lightning?
        print('Load model from checkpoint')
        cfg, model, loaded_params = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
        cfg.general.pretrained_params = loaded_params

    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    callbacks = []
    for cb in cfg.callbacks:
        # create the callback object
        cb_obj = hydra.utils.instantiate(cb)

        if type(cb_obj) == pytorch_lightning.callbacks.ModelCheckpoint:
            if cfg.general.no_ckpt:
                print('Checkpointing disabled, dont add callback')
                continue
            # defaults:
            # monitor: val_mean_ap_50
            # filename: "{epoch}-{val_mean_ap_50:.3f}"
            if cfg.general.gen_captions or cfg.general.gen_part_captions:
                if cfg.general.dont_eval_on_val and cfg.general.eval_on_train > 0:
                    caption_ckpt_split = 'train'
                else:
                    caption_ckpt_split = 'val'

                print(f'Use {caption_ckpt_split} cider for checkpointing')
                # if captioning, use caption score for checkpoint?
                if cfg.general.gen_captions:
                    cb_obj.monitor = f"{caption_ckpt_split}_cider"
                elif cfg.general.gen_part_captions and not cfg.general.gen_captions: # part caps only
                    cb_obj.monitor = f"{caption_ckpt_split}_part_cider"
                cb_obj.filename = f'{{epoch}}{{{cb_obj.monitor}:.3f}}'

            elif cfg.general.dont_eval_on_val and cfg.general.eval_on_train > 0: 
            # not evaluating on val -> use train ap 50 for checkpointing
                print('Use train ap 50 for checkpointing')
                cb_obj.monitor = "train_mean_ap_50"
                cb_obj.filename = "{epoch}-train_ap50_{train_mean_ap_50:.3f}"
        elif type(cb_obj) == pytorch_lightning.callbacks.LearningRateMonitor and cfg.general.no_log:
            print('Cant log LR when no_log is True, skipping')
            continue
        print('Adding callback:', cb_obj)
        callbacks.append(cb_obj)

    if not cfg.general.no_ckpt:
        # always do regular ckpting
        cb_obj = RegularCheckpointing()
        print('Adding callback:', cb_obj)
        callbacks.append(cb_obj)

    runner = Trainer(
        enable_checkpointing=not cfg.general.no_ckpt,
        logger=loggers,
        devices=cfg.general.gpus,
        callbacks=callbacks,
        # weights_save_path=str(cfg.general.save_dir), # pl 1.7.0
        default_root_dir=str(cfg.general.save_dir), # pl 1.8.0
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        devices=cfg.general.gpus,
        logger=loggers,
        # weights_save_path=str(cfg.general.save_dir), # pl 1.7.0
        default_root_dir=str(cfg.general.save_dir), # pl 1.8.0
        **cfg.trainer,
    )
    runner.test(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
