"""
    Bounding box intersection over union calculation.
    Borrowed from pytorch SSD implementation : https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
"""

import torch, sys
import numpy as np
from matplotlib import pyplot as plt


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if len(box_a.shape) != len(box_b.shape):
        print("please provide the uniform bounding box coordinates")
        sys.exit(1)
    if len(box_a.shape) == 1 and len(box_b.shape) == 1:
        box_a = box_a.view(1, -1)
        box_b = box_b.view(1, -1)
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def compute_IoU(pred_bb, gt_bb, pred_conf, confidence_threshold):
    """
    Compute the intersection over Union and return the calculated value
    if above confidence threshold else zero
    Args:
        pred_bb: Predicited Bounding Box coordinates(xmin, ymin, xmax, ymax)
        gt_bb: Ground truth Bounding Box coordinates(xmin, ymin, xmax, ymax)
        pred_conf: predicted confidence
        confidence_threshold: confidence threshold to consider

    Returns: Computed IoU
    """
    IoU = jaccard(pred_bb, gt_bb)
    IoU[pred_conf < confidence_threshold, :] = 0
    return IoU


def get_precision_recall(pred_bb, gt_bb, IOUThreshold=0.5):
    """
    Calculate the precision and recall value
    Args:
        pred_bb: Predicited Bounding Box coordinates(xmin, ymin, xmax, ymax)
        gt_bb: Ground truth Bounding Box coordinates(xmin, ymin, xmax, ymax)
        IOUThreshold: IoU threshold value

    Return: tuple of precision and recall values
    """
    pred_bb = torch.tensor(pred_bb)
    gt_bb = torch.tensor(gt_bb)
    TP = np.zeros(len(pred_bb))
    FP = np.zeros(len(pred_bb))
    gt_index = np.zeros(len(pred_bb))

    for p in range(len(pred_bb)):
        iouMax = sys.float_info.min
        gtMax = -1
        for gt in range(len(gt_bb)):

            iou = jaccard(pred_bb[p], gt_bb[gt])
            if iou > iouMax:
                iouMax = iou
                gtMax = gt
        gt_index[p] = gtMax
        if iouMax >= IOUThreshold:
            TP[p] = 1
        else:
            FP[p] = 1

    acc_TP = np.cumsum(TP)
    acc_FP = np.cumsum(FP)
    total_gt_bbox = len(gt_bb)

    recall = acc_TP / total_gt_bbox
    precision = np.divide(acc_TP, (acc_TP + acc_FP))
    return precision, recall


def plot_precision_recall_curve(precision, recall):
    """
    Plot the precision recall curve
    Args:
        precision: A list of precision value
        recall:  A list of recall value
    Return: None
    """
    plt.plot(recall, precision, label='Precision')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(shadow=True)
    plt.grid()
    plt.show()
