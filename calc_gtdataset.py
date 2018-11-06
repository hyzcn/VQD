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
        sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx = data 
        idxs.extend(sent_id.tolist())        
        true.extend(gtbox.tolist())  
        B = len(box_coordsorig)               
        iipred = torch.cat( (torch.tensor(range(0,B)).unsqueeze(1).long(),idx.long()),dim=1)
        predbox = box_coordsorig[iipred[:,0],iipred[:,1]]
        pred.extend(predbox.tolist())
    
    
    traingt = torch.tensor(true)
    trainpred = torch.tensor(pred)
    trainacc = eval_extra.getaccuracy(traingt,trainpred)
    print("\tAccuracy {:.2f}%".format(trainacc))
    
    
#%%
"""
Only 91.13%  of the actual COCO gt is present in the bottomup 
features.

"""    