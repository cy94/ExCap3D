# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


import os
import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from utils.votenet_utils.metric_util import calc_iou  # axis-aligned 3D box IoU


def get_iou(bb1, bb2):
    """Compute IoU of two bounding boxes.
    ** Define your bod IoU function HERE **
    """
    # pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d


from box_util import box3d_iou


def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


# eval for a single class
def eval_det_cls(
    pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for a single class.
    Input:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0

    # each scene in GT
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox) 
        # track the IOU and box ID as well
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": det}

    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # construct dets
    # scene id of each box
    image_ids = []
    # confidence of each box
    confidence = []
    # each box
    BB = []
    for img_id in pred.keys():
        # go over each pred box
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence -> across all scene ids, sorts within each scene as well
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    # total num pred boxes
    nd = len(image_ids)
    # mark each pred box if its TP or FP
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # for each pred box (across all scenes)
    for d in range(nd):
        # GT data for this scene
        R = class_recs[image_ids[d]]
        # current pred bounding box
        bb = BB[d, ...].astype(float)
        # max overlap till now for this pred box?
        ovmax = -np.inf
        # gt bbox for this scene
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # go over each GT bbox
            for j in range(BBGT.shape[0]):
                # j = GT box index
                # iou of pred box and GT box
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    # track the GT to which this pred box has max iou and should be assigned
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            # nothing was assigned to this GT box
            if not R["det"][jmax]:
                # mark as TP
                tp[d] = 1.0
                # mark that this GT box has a pred box
                R["det"][jmax] = 1
            else:
                # mark as FP
                fp[d] = 1.0
        else:
            # pred didnt overlap with any GT box, its an FP
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(
        pred, gt, ovthresh, use_07_metric, get_iou_func
    )
    return (rec, prec, ap)


def eval_det(
    pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}

    # store gt, pred for each class
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            # JONAS ADAPTATION TODO
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            # =====================
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    # compute for each class
    for classname in gt.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls(
            pred[classname],
            gt[classname],
            ovthresh,
            use_07_metric,
            get_iou_func,
        )

    return rec, prec, ap


from multiprocessing import Pool


def eval_det_multiprocessing(
    pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(
        eval_det_cls_wrapper,
        [
            (
                pred[classname],
                gt[classname],
                ovthresh,
                use_07_metric,
                get_iou_func,
            )
            for classname in gt.keys()
            if classname in pred
        ],
    )
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])

    return rec, prec, ap
