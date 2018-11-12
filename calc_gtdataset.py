import numpy as np
import torch
import eval_extra
from data import ReferDataset
from torch.utils.data import  DataLoader
from tqdm import tqdm
"""
Find how many of our calculated either frcnn 101 or bottom up feats 
have IOU>0.5 with the gt of the mscoco for the images in all splits.
 """
if __name__ == "__main__":
    import config   
    ds = 'refcocog'
    config.global_config['dictionaryfile'] = config.global_config['dictionaryfile'].format(ds)
    config.global_config['glove'] = config.global_config['glove'].format(ds)      
    dataloader_kwargs = {}
    dataloader_kwargs = {**config.global_config , **config.dataset[ds] }
    dataloader_kwargs['split'] = 'train'
    testds = ReferDataset(istrain=False,**dataloader_kwargs)
    B = 64
    loader = DataLoader(testds,shuffle=False, num_workers = 0,batch_size=B)
    true = []
    pred = []
    idxs = []

    for i,data in enumerate(tqdm(loader)):
        sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx,correct = data
        idxs.extend(sent_id.tolist())        
        true.extend(gtbox.tolist())  
        B = len(box_coordsorig)               
        iipred = torch.cat( (torch.tensor(range(0,B)).unsqueeze(1).long(),idx.long()),dim=1)
        predbox = box_coordsorig[iipred[:,0],iipred[:,1]]
        pred.extend(predbox.tolist())
    
    
    traingt = torch.tensor(true)
    trainpred = torch.tensor(pred)
    trainacc = eval_extra.getaccuracy(traingt,trainpred)
    print("\nAccuracy using bottomup {:.2f}%".format(trainacc))
    
"""
Only 91.13%  of the actual COCO gt is present in the bottomup 
features.

"""    

#%%

import json
from collections import defaultdict
js = json.load(open('/media/manoj/hdd/VQD/MAttNet/detections/refcocog_umd/res101_coco_minus_refer_notime_dets.json'))

qid2ent = defaultdict(list)
for ent in js:
    qid2ent[ent['image_id']].append(ent)
print ("total detections",len(qid2ent))
for kid in qid2ent:
    print (kid,len(qid2ent[kid]))
    
    
#%%
    
    # box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

acc = 0
L = len(testds.data)
for ent in testds.data:
    imgid = ent['image_id']
    gtbox_xywh = np.array([ent['gtbox']])
    boxes_xywh = np.array([ b['box'] for b in qid2ent[imgid]])    
    gtbox_xyxy = torch.from_numpy(xywh_to_xyxy(gtbox_xywh))
    boxes_xyxy = torch.from_numpy(xywh_to_xyxy(boxes_xywh))
    ious = eval_extra.getIOU(gtbox_xyxy,boxes_xyxy)> 0.5
    iou = ious.sum().item()
    if iou>=1:
        acc +=1.0

print("\nAccuracy using Mattnet Boxes {:.2f}%".format(100*acc/L))   
