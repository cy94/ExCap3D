import itertools
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics.text.bert import BERTScore
from nltk import word_tokenize
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from codetiming import Timer

from benchmark.cider import Cider

def eval_cap_exact_match(preds, gts):
    '''
    preds: list of pred strings
    gts: list of gt strings
    '''
    scores = np.zeros(len(preds))
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        scores[i] = int(pred == gt)

    return scores

def plot_cap_eval(cap_data_with_scores, eval_dict, out_path, class_list):
    for metric in eval_dict['per_gt_class_scores']:
        vals = []
        for classname in class_list:
            vals.append(eval_dict['per_gt_class_scores'][metric].get(classname, 0))
        xcoords = list(range(len(class_list)))
        # bar plot
        plt.figsize = (10, 6)
        plt.bar(xcoords, vals)
        plt.xticks(xcoords, class_list)
        plt.xlabel('Class')
        plt.ylabel(metric)
        plt.title(f'{metric} per class')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        plot_out_path = out_path / f'{metric}.png'
        plot_out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_out_path)
        plt.close()

    gt_counter, pred_counter = Counter(), Counter()
    # unique gts and preds
    for scene_id in tqdm(cap_data_with_scores, desc='scene'):
        for obj_id in tqdm(cap_data_with_scores[scene_id], desc='obj', leave=False):
            gt = cap_data_with_scores[scene_id][obj_id]['gt']
            pred = cap_data_with_scores[scene_id][obj_id]['pred']
            gt_counter.update(gt) # gt is a list
            pred_counter.update([pred])

    top_n = 40
    gt_top_n = gt_counter.most_common(top_n)
    labels = [k for (k, _) in gt_top_n]
    labels = [s.replace('$', '') for s in labels]
    plot_bar(labels, [v for (_, v) in gt_top_n], 'GT caption', 'Count', 'GT caption distribution', figsize=(12, 8), rot=30)
    plot_out_path = out_path / 'gt_caption_dist.png'
    plot_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out_path); plt.clf(); plt.close()

    pred_top_n = pred_counter.most_common(top_n)
    # cant have special characters in labels!
    labels = [k for (k, _) in pred_top_n]
    labels = [s.replace('$', '') for s in labels]
    plot_bar(labels, [v for (_, v) in pred_top_n], 'Pred caption', 'Count', 'Pred caption distribution', figsize=(12, 8), rot=30)
    plot_out_path = out_path / 'pred_caption_dist.png'
    plot_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out_path); plt.clf(); plt.close()


def plot_bar(labels, counts, xlabel, ylabel, title, figsize=(10,6),rot=30):
    xcoords = list(range(len(labels)))
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.bar(xcoords, counts, align='center')
    ax.set_xticks(xcoords)
    ax.set_xticklabels(labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rot, ha='right')
    plt.tight_layout()

def eval_assigned_captions(assigned_captions, eval_against_gt=False, n_top_bottom=20,
                           eval_bertscore=False, eval_bleu_small=False):
    '''
    writes into the original dict - make a copy outside if this needs to be preserved
    '''
    # do these always, fast
    rouge = ROUGEScore()
    bleu4 = BLEUScore()
    cider = Cider()

    if eval_bleu_small: # not very useful
        bleu1 = BLEUScore(n_gram=1)
        bleu2 = BLEUScore(n_gram=2)
        bleu3 = BLEUScore(n_gram=3)

    if eval_bertscore: # takes a lot of time
        bertscore = BERTScore(device='cuda')

    sample_scores = defaultdict(list)
    # track scores for each gt sem class
    # gt sem class -> metric -> avg score
    keys, preds, gts = [], [], []
    gt_sem_labels, pred_sem_labels = [], []
    gt_per_class, preds_per_class = defaultdict(list), defaultdict(list)

    num_scenes = 0
    
    # float dict for logging / nested dict with everything
    avg_scores, eval_dict = {}, {}

    for scene_id in tqdm(assigned_captions, desc='scene'):
        num_scenes += 1

        for obj_id in tqdm(assigned_captions[scene_id], desc='obj', leave=False):
            gt = assigned_captions[scene_id][obj_id]['gt'] # gt is a list of captions
            if eval_against_gt:
                pred = gt[0] 
            else:
                pred = assigned_captions[scene_id][obj_id]['pred'] # single pred

            keys.append((scene_id, obj_id))
            preds.append(pred)
            gts.append(gt) # list of gt

            gt_sem_label = assigned_captions[scene_id][obj_id]['gt_sem_label']
            gt_sem_labels.append(gt_sem_label)

            if eval_against_gt:
                pred_sem_label = gt_sem_label
            else:
                pred_sem_label = assigned_captions[scene_id][obj_id]['pred_sem_label']
            pred_sem_labels.append(pred_sem_label)

            gt_per_class[gt_sem_label].append(gt)
            preds_per_class[pred_sem_label].append(pred)

            # cider calculated over whole corpus
            this_sample_scores = {
                'rouge': rouge([pred], [gt])['rougeL_fmeasure'],
                'bleu4': bleu4([pred], [gt]),
                'meteor': meteor_score([word_tokenize(gt_cap) for gt_cap in gt], word_tokenize(pred)),
                'exact_match': float(pred in gt),
                'sem_label_exact_match': float(pred_sem_label == gt_sem_label)
            }
            if eval_bleu_small:
                this_sample_scores.update({
                    'bleu1': bleu1([pred], [gt]),
                    'bleu2': bleu2([pred], [gt]),
                    'bleu3': bleu3([pred], [gt]),
               })

            for metric in this_sample_scores:
                assigned_captions[scene_id][obj_id][metric] = float(this_sample_scores[metric])   
                sample_scores[metric].append(this_sample_scores[metric])

    print(f'Evaluated {num_scenes} scenes')

    ######## add missing per sample scores #########
    # bert score
    if eval_bertscore:
        # TODO: doesnt support multiple GT! use the official bertscore library
        # currently pick the first gt caption for each sample
        with Timer(name='bertscore', text="{name} done in {:.4f} seconds"):
            bert_scores = bertscore(preds, [gt[0] for gt in gts])['f1']
            sample_scores['bert'] = bert_scores
            for (key, sample_bert) in zip(keys, bert_scores):
                scene_id, obj_id = key
                assigned_captions[scene_id][obj_id]['bert'] = float(sample_bert)

    # cider 
    cider_gts = {f'{key[0]}_{key[1]}': gt for key, gt in zip(keys, gts)}
    cider_preds = {f'{key[0]}_{key[1]}': [pred] for key, pred in zip(keys, preds)}
    cider_avg, cider_scores = cider.compute_score(cider_gts, cider_preds)
    sample_scores['cider'] = cider_scores

    for (key, sample_cider) in zip(keys, cider_scores):
        scene_id, obj_id = key
        assigned_captions[scene_id][obj_id]['cider'] = float(sample_cider)
    ##########################################################
    ########## aggregate metrics + logged metrics ############ 
    # for each metric, get the top10 and bottom10 samples
    for metric in sample_scores:
        sort_ndx = np.argsort(sample_scores[metric])

        # copy the the whole obj data for these to assigned_captions['top/bottom']
        # sorted in worst to best -> reverse it to get best to worst
        top_keys = [keys[i] for i in sort_ndx[-n_top_bottom:][::-1]]
        eval_dict['top_'+metric] = {f'{key[0]}_{key[1]}': assigned_captions[key[0]][key[1]] for key in top_keys}

        # already sorted in worst to best
        bottom_keys = [keys[i] for i in sort_ndx[:n_top_bottom]]
        eval_dict['bottom_'+metric] = {f'{key[0]}_{key[1]}': assigned_captions[key[0]][key[1]] for key in bottom_keys}
    ######### finalize scores for logging #########
    avg_scores.update({k: float(torch.Tensor(v).mean()) for k, v in sample_scores.items()})
    # cider average from cider output
    avg_scores['cider'] = cider_avg
    if eval_bertscore:
        avg_scores['bert'] = float(torch.Tensor(bert_scores).mean())

    # per GT sem class results
    # metric -> gt class -> score
    per_gt_class_scores = defaultdict(dict)

    for gt_sem_label in set(gt_sem_labels):
        # samples with this gt label
        ndx = [i for i, label in enumerate(gt_sem_labels) if label == gt_sem_label]
        for metric in sample_scores.keys():
            avg_score = np.mean([sample_scores[metric][i] for i in ndx])    
            per_gt_class_scores[metric][gt_sem_label] = float(avg_score)

    eval_dict['per_gt_class_scores'] = per_gt_class_scores

    # other computed avg scores -> dont use these again
    all_gt_caps = list(itertools.chain.from_iterable(gts))
    avg_scores['unique_gt_cap_frac'] = len(set(all_gt_caps)) / len(all_gt_caps)
    avg_scores['unique_pred_cap_frac'] = len(set(preds)) / len(preds)
    eval_dict['num_unique_gt'] = len(set(all_gt_caps))
    eval_dict['num_unique_pred'] = len(set(preds))

    num_total_caps = len(all_gt_caps)
    print(f'Evaluated {num_total_caps} captions')
    # store 
    eval_dict['num_total_caps'] = num_total_caps
    num_empty_preds = len([pred for pred in preds if pred == ''])
    eval_dict['num_empty_preds'] = num_empty_preds
    print(f'Empty preds: {num_empty_preds}')

    sem_match = np.array(sample_scores['sem_label_exact_match'])
    sem_correct, sem_wrong = sem_match == 1, sem_match == 0

    # metrics over sem correct, sem wrong
    for metric in sample_scores:
        sample_scores[metric] = np.array(sample_scores[metric])
        eval_dict[f'{metric}_sem_correct'] = float(np.mean(sample_scores[metric][sem_correct]))
        eval_dict[f'{metric}_sem_wrong'] = float(np.mean(sample_scores[metric][sem_wrong]))

    # for each class, dont log, put in eval_dict
    for sem_label, gts_for_class in gt_per_class.items():
        all_gt_caps_class = list(itertools.chain.from_iterable(gts_for_class))
        eval_dict[f'unique_gt_cap_frac_{sem_label}'] = len(set(all_gt_caps_class)) / len(all_gt_caps_class)
    for sem_label, preds_for_class in preds_per_class.items():
        eval_dict[f'unique_pred_cap_frac_{sem_label}'] = len(set(preds_for_class)) / len(preds_for_class)

    # everything from avg scores into eval dict
    eval_dict['caption_scores'] = {k: float(v) for k, v in avg_scores.items()}


    return avg_scores, assigned_captions, eval_dict