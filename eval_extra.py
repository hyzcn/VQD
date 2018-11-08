import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

def getIOU(b_ij, b_ji):    
    """Finds Iou between each entries in two tensors.
    Either of b_ij or b_ji should have only one box of shape (1,4)
    and other can have boxes of size (N,4).
    Or both should have same number of rows
  
    ----------
    returns iou as a tensor
    """

    area_ij = (b_ij[..., 2] - b_ij[..., 0]) * (b_ij[..., 3] - b_ij[..., 1])
    area_ji = (b_ji[..., 2] - b_ji[..., 0]) * (b_ji[..., 3] - b_ji[..., 1])

    righmost_left = torch.max(b_ij[..., 0], b_ji[..., 0])
    downmost_top = torch.max(b_ij[..., 1], b_ji[..., 1])
    leftmost_right = torch.min(b_ij[..., 2], b_ji[..., 2])
    topmost_down = torch.min(b_ij[..., 3], b_ji[..., 3])

    # calucate the separations
    left_right = (leftmost_right - righmost_left)
    up_down = (topmost_down - downmost_top)

    # don't multiply negative separations,
    # might actually give a postive area that doesn't exit!
    left_right = torch.max(0*left_right, left_right)
    up_down = torch.max(0*up_down, up_down)
    overlap = left_right * up_down
    iou = overlap / (area_ij + area_ji - overlap)
    iou = iou.unsqueeze(-1)
    return iou
   
    

def getaccuracy(gt, pred):
    """Finds performance metrics like MattNet.
    Calcultes Ious for each prediciton and gt boxes tensors.
    Iou > 0.5 is a correct comprehension and hence value of 1
  
    ----------
    returns accuracy value
    """
    iou = getIOU(gt,pred)
    correct = iou > 0.5
    correct = correct.sum().item()    
    acc = 100.0 * correct / len(iou)
    return acc
    
    

def compute_overall(predictions):
  """
  check precision and recall for predictions.
  Input: predictions = [{ref_id, cpts, pred}]
  Output: overall = {precision, recall, f1}
  """
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for item in predictions:
    cpts, pred = item['gd_att_wds'], item['pred_att_wds']
    inter = list(set(cpts).intersection(set(pred)))
    # add to overall 
    NC += len(inter)
    NP += len(pred)
    NR += len(cpts)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall'])
  return overall

def convert_xywh_x1y1x2y2(box_coords):
    return torch.cat( (box_coords[:,0:2] , box_coords[:, 0:2] + box_coords[:, 2:4] - 1),dim=1)

# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


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

            iou = getIOU(pred_bb[p], gt_bb[gt])
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

#%%
    
if __name__ == "__main__":
    
    box_coords = [[  0.9600,   0.4800, 252.5600, 356.6700],
                [285.7800,   0.0000,  47.0500,  70.6300],
                [151.9600, 139.4600, 454.9300, 283.7300],
                [254.8600, 208.3600,  47.6300,  40.3800],
                [468.3000,   0.9100, 171.7000, 116.1200],
                [314.5100, 352.9800,  64.4400,  40.4000],
                [321.1600, 219.3300,  50.3800,  34.7400],
                [326.9400, 248.8300,  25.7100,  23.7300],
                [362.5700, 277.1000,  58.5600,  49.9700],
                [412.1400, 149.1200,  56.5200,  40.6100],
                [356.1300, 145.4800,  44.8600,  43.6400],
                [443.0100, 221.9500,  47.0800,  28.3500],
                [528.9900, 270.2700,  53.8600,  40.3900],
                [261.7600, 252.2100,  60.0600,  38.2200],
                [117.1200,   0.5400, 344.0200, 180.5900],
                [391.9400, 345.0400, 105.6200,  49.8800],
                [320.9600, 321.7600,  60.7600,  40.5100],
                [  1.1100,   1.1100, 635.8900, 420.2200],
                [240.3400, 174.1600, 106.9500,  93.4700],
                [263.0000, 253.0000, 151.0000,  55.0000]]
    
    box_coords = torch.tensor(box_coords)
    gtbox  = [[468.3000,   0.9100, 171.7000, 116.1200]]
    gtbox = torch.tensor(gtbox)
    #convert xywh to x1 y1 x2 y2 format
    boxes =  convert_xywh_x1y1x2y2(box_coords)
    gt = convert_xywh_x1y1x2y2(gtbox)
    ious = getIOU(gt,boxes)
    print (ious)
    
    print ("if you wnat to get a matrix then just do this.")
    gtbox  = [[468.3000,   0.9100, 171.7000, 116.1200],
              [468.3000,   0.9100, 171.7000, 116.1200]]
    
    gtbox = torch.tensor(gtbox)
    gt = convert_xywh_x1y1x2y2(gtbox)
    ious = getIOU(gt.unsqueeze(1),boxes)
    print (ious)