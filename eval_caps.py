'''
read caps output and eval 
exact match scores of captions and pred sem label vs gt sem label
other caption metrics
get best and worst captions by metric
'''
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path
from codetiming import Timer

from benchmark.evaluate_caption import eval_assigned_captions, plot_cap_eval
from scannetpp.common.file_io import load_json, read_txt_list, write_json

def removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[3:]
    return s

def removesuffix(s, suffix):
    if s.endswith(suffix):
        return s[:-3]
    return s

def remove_sos_eos_from_preds(preds):
    for scene_id, scene_preds in preds.items():
        for objid, obj_data in scene_preds.items():
            pred = obj_data['pred']

            pred = removeprefix(pred, 'sos').strip()
            pred = removesuffix(pred, 'eos').strip()

            preds[scene_id][objid]['pred'] = pred

            gts = obj_data['gt']
            if type(gts) != list:
                gts = [gts]
            new_gts = []
            # do the same for each gt
            for gt in gts:
                gt = removeprefix(gt, 'sos').strip()
                gt = removesuffix(gt, 'eos').strip()
                new_gts.append(gt)
            preds[scene_id][objid]['gt'] = new_gts

    return preds

def read_gt_caps_to_assigned_format(caps_dir, scene_list, caption_key, assigned_caps=None):
    '''
    read GT from the GT caption files for all objects
    '''
    caps_dir = Path(caps_dir)
    # sceneid -> objid -> pred str (empty), gt list, gt_sem_label (dummy), pred_sem_label (dummy)
    if assigned_caps is None:
        print(f'Create empty assigned caps and read GT from GT dir')
        assigned_caps = {}
        create_new = True
    else:
        create_new = False
        print(f'Overwrite existing GT in assigned caps with GT from GT dir')

    for scene_id in tqdm(scene_list):
        cap_data = load_json(caps_dir / f'{scene_id}.json')

        if create_new:
            # create empty entry
            assigned_caps[scene_id] = {}

        for objid, obj_data in cap_data['objects'].items():
            # all objects dont have GT part captions
            if caption_key not in obj_data:
                continue
            cap = obj_data[caption_key]
            # keep the first gt
            if type(cap) == list:
                cap = [cap[0]]
            if type(cap) != list:
                cap = [cap]

            if objid not in assigned_caps[scene_id]:
                assigned_caps[scene_id][objid] = {}
            # if it already exists, overwrite, otherwise create new entries
            assigned_caps[scene_id][objid].update({
                'gt': cap,
                'gt_sem_label': 'dummy',
                'pred_sem_label': 'dummy'
            })

    return assigned_caps

def read_pred_caps_gt_format(pred_caps_dir, assigned_caps, instance_preds_dir, caption_key):
    '''
    read "predictions" that are in the same format as the GT 
    if instance preds dir is provided, read IOUs and use empty preds for objects with IOU < 0.5
    '''
    pred_caps_dir = Path(pred_caps_dir)

    for scene_id in assigned_caps:
        pred_caps = load_json(pred_caps_dir / f'{scene_id}.json')

        # filter out instances with IOU < 0.5
        if instance_preds_dir:
            instance_info = load_json(Path(instance_preds_dir) / f'{scene_id}.json')
        else:
            instance_info = None

        for objid in assigned_caps[scene_id]:
            if (instance_info and objid not in instance_info) or objid not in pred_caps['objects']:
                # didnt have an assigned pred for this object
                # or didnt generate a caption for this object (visibility, etc)
                pred = ''
            else:
                # filter by iou and get the pred
                if instance_info:
                    obj_info = instance_info[objid]
                    iou = obj_info['iou']
                    if iou < 0.5:
                        pred = ''
                    else:
                        pred = pred_caps['objects'][objid][caption_key]
                else:
                    pred = pred_caps['objects'][objid][caption_key]

            # if list, keep first pred
            if type(pred) == list:
                pred = pred[0]
            assigned_caps[scene_id][objid]['pred'] = pred

    return assigned_caps

def caption_str_to_tokens(caption_str):
        return caption_str.lower().replace('.', ' . ').replace(',', ' , ').split()

def tokenize_and_join_gt(assigned_caps):
    for scene_id in assigned_caps:
        for objid in assigned_caps[scene_id]:
            tokens_lists = [caption_str_to_tokens(gt) for gt in assigned_caps[scene_id][objid]['gt']]
            # join each list of tokens to get multiple GT 
            assigned_caps[scene_id][objid]['gt'] = [' '.join(tokens) for tokens in tokens_lists]
    return assigned_caps

@hydra.main(config_path="conf", config_name="eval_caps")
def main(cfg: DictConfig):
    if cfg.pred_caps_dir_gt_format:
        print(f'Reading GT caps from GT dir')
        scene_list = read_txt_list(cfg.scene_list_file)
        # read GT from GT dir
        assigned_caps_only_gt = read_gt_caps_to_assigned_format(cfg.gt_caps_dir, scene_list, cfg.gt_caption_key)
        print(f'Reading pred caps from pred (generated) dir')
        # read caps in the assigned caps format + filter by IOU from the instance preds dir
        assigned_caps = read_pred_caps_gt_format(cfg.pred_caps_dir_gt_format, assigned_caps_only_gt, cfg.instance_preds_dir, cfg.gt_caption_key)
    else:
        print('Reading pred+GT assigned caps from pred path:', cfg.cap_pred_file)
        assigned_caps = load_json(cfg.cap_pred_file)['preds'] # ignore the scores that are already there, new eval

        if cfg.use_different_gt_from_dir:
            print(f'Replace GT caps with GT from GT dir')
            scene_list = read_txt_list(cfg.scene_list_file)
            assigned_caps = read_gt_caps_to_assigned_format(cfg.gt_caps_dir, scene_list, cfg.gt_caption_key, assigned_caps=assigned_caps)

    if cfg.remove_sos_eos:
        # remove sos and eos from begin/end of each prediction
        assigned_caps = remove_sos_eos_from_preds(assigned_caps)

    if cfg.tokenize_and_join_gt:
        # pq3d format - lowercase, tokenize and add space, then join tokens
        # to make it consistent with the training data for that method
        assigned_caps = tokenize_and_join_gt(assigned_caps)

    with Timer():
        cap_scores, assigned_caps_with_scores, eval_dict = eval_assigned_captions(assigned_caps, eval_against_gt=cfg.eval_against_gt,
                                                                                  eval_bertscore=False)
    print('Avg scores:', cap_scores)

    # combine to single dict2
    caption_output = {
        'preds': assigned_caps_with_scores,
        'eval': eval_dict
    }

    if cfg.output_file:
        print('Writing caps eval to', cfg.output_file)
        # make parent
        Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg.output_file, caption_output)

        if cfg.class_list_file:
            print(f'')
            class_list = read_txt_list(cfg.class_list_file)
            plot_cap_eval(assigned_caps_with_scores, eval_dict,  Path(cfg.output_file).parent / 'caption_eval_plots', class_list)

if __name__ == '__main__':
    main()