from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from models.dictionary import Dictionary
from models.language import tokenize_ques
import os.path as osp
import json
from eval_extra import getIOU,convert_xywh_x1y1x2y2
import pickle
#"""
#{'ann_id': 1706357,
# 'category_id': 1,
# 'file_name': 'COCO_train2014_000000421086_4.jpg',
# 'image_id': 421086,
# 'ref_id': 14024,
# 'sent_ids': [39906, 39907, 39908],
# 'sentences': [{'raw': 'left white shirt',
#   'sent': 'left white shirt',
#   'sent_id': 39906,
#   'tokens': ['left', 'white', 'shirt']},
#  {'raw': 'white shirt',
#   'sent': 'white shirt',
#   'sent_id': 39907,
#   'tokens': ['white', 'shirt']},
#  {'raw': 'top left corner: apron strings',
#   'sent': 'top left corner apron strings',
#   'sent_id': 39908,
#   'tokens': ['top', 'left', 'corner', 'apron', 'strings']}],
# 'split': 'testA'} """


class ReferDataset(Dataset):

    def __init__(self,**kwargs):
    
        dataset = kwargs.get('dataset')
        splitBy = kwargs.get('splitBy')
        split = kwargs.get('split')

        
        data_json = osp.join('cache/prepro', dataset +"_"+ splitBy , split +'.json')
        
        with open(data_json,'r') as f:
            self.data = json.load(f)


        dictfile = kwargs.get('dictionaryfile')
        self.dictionary = Dictionary.load_from_file(dictfile)    
        if kwargs.get('testrun'):
            self.data = self.data[:32]
            
        self.spatial = True            
        print ("Dataset [{}] loaded....".format(dataset,split))
        print ("Split [{}] has {} ref exps.".format(split,len(self.data)))

    def _process_boxes(self,bboxes,image_w,image_h):
            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h
            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]      
            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)  
                
            return spatial_features  
       
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ent = self.data[idx]
        sent_id = ent['sentence']['sent_id']
        file_name = ent['image_info']['file_name']
        img_id = ent['image_id']
        ans = ent['category_id']  
        W = ent['image_info']['width']
        H = ent['image_info']['height']
        que = ent['sentence']['sent']                  
        gtbox = ent['gtbox']
        gtbox = torch.tensor(gtbox)
        #boxes from refcoc is in xywh format
        gtboxorig = convert_xywh_x1y1x2y2(gtbox.unsqueeze(0)).squeeze(0)
        box_coords = ent['boxes']
        box_coords = torch.tensor(box_coords)
        
        
        if 'train' in file_name:
            pk = pickle.load(open(osp.join("feats","train2014",file_name+".pkl"),"rb"))
        elif 'val' in file_name:
            pk = pickle.load(open(osp.join("feats","val2014",file_name+".pkl"),"rb"))    
            
        L =  len(pk) - 1 # lenght of entries in pickle file

        if L<=0:
            L = 1           
            
        N =  20  # maximum entries to use
        L = 20   # uncomment if you want variable L 
        if L>N:
            L=N   
            
        box_feats = np.zeros((N,2048),dtype=np.float32)    
        box_coords = np.zeros((N,4))
        for i,ent in enumerate(pk[:-1]):
            if i == N:
                break
            box_feats[i,:] = ent['feat'].flatten()
            box_coords[i,:] = np.array(ent['coords'])
            
        lastent = pk[-1]
        wholefeat = lastent['image']
        W = lastent['w']
        H = lastent['h']            
            
        box_feats = torch.from_numpy(box_feats)
        box_coordsorig = box_coords
        box_coords_6d = self._process_boxes(box_coords,W,H)       
        box_coords_6d  = torch.from_numpy(box_coords_6d)
        
        #boxes in h4files are in x1 y1 x2 y2 format
        iou = getIOU(gtboxorig.unsqueeze(0),torch.from_numpy(box_coordsorig).float())
        correct = iou>0.5
        _,idx = torch.max(iou,dim=0)
#        print (iou,iou.shape,box_coordsorig,"index",idx)
        gtboxiou = box_coordsorig[idx]
        gtboxiou = torch.from_numpy(gtboxiou)
        
        tokens = tokenize_ques(self.dictionary,que)
        qfeat = torch.from_numpy(tokens).long()
        Lvec = torch.zeros(N).long()
        Lvec[:L] = 1        
        return sent_id,ans,box_feats,box_coordsorig,box_coords_6d.float(),\
                    gtboxorig.float(),qfeat,Lvec,idx,correct.view(-1)

#%%
if __name__ == "__main__":
    import config   
    ds = 'refcoco+'
    config.global_config['dictionaryfile'] = config.global_config['dictionaryfile'].format(ds)
    config.global_config['glove'] = config.global_config['glove'].format(ds)      
    dataloader_kwargs = {}
    dataloader_kwargs = {**config.global_config , **config.dataset[ds] }
    dataloader_kwargs['split'] = 'val'
    cd = ReferDataset(**dataloader_kwargs)
    it = iter(cd)
#%%
    data =  next(it)
    print (data)
    sent_id,ans,box_feats,box_coordsorig,box_coords_6d,\
            gtbox,qfeat,L,idx,correct = data
      
  