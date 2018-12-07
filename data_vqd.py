from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from models.dictionary import Dictionary
from models.language import tokenize_ques
import os.path as osp
import json
from eval_extra import getIOU,convert_xywh_x1y1x2y2



# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

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
        self.image_features_path_coco = kwargs.get('vqd_detfeats').format(split)
        self.coco_id_to_index =  self.id_to_index(self.image_features_path_coco)  
        print ("Dataset [{}] loaded....".format(dataset,split))
        print ("Split [{}] has {} ref exps.".format(split,len(self.data)))
        
        
        
        #only use the questinos having 1 bbox as answer
        datanew = []
        for ent in self.data:
            #some image ids are not in the dataset
            if ent['image_id'] in self.coco_id_to_index:
                gtbox = ent['gtbox']
                if len(gtbox[0]) != 0  and len(gtbox) == 1:
                    datanew.append(ent)
        self.data = datanew
        

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


    def id_to_index(self,path):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
               
        with  h5py.File(path, 'r') as features_file:
            coco_ids = features_file['ids'][:]
        coco_id_to_index = {name: i for i, name in enumerate(coco_ids)}
        return coco_id_to_index       
        
     
    def _load_image_coco(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path_coco, 'r')
           
        index = self.coco_id_to_index[image_id]
        L = self.features_file['num_boxes'][index]
        W = self.features_file['widths'][index]
        H = self.features_file['heights'][index]
        box_feats = self.features_file['features'][index]
        box_locations = self.features_file['boxes'][index]
        #is in xywh format
        box_locations = xywh_to_xyxy(box_locations)
        score = self.features_file['scores'][index]
        category_id = self.features_file['catids'][index]       
        # find the boxes with all co-ordinates 0,0,0,0
        #L = np.where(~box_locations.any(axis=1))[0][0]
                
        if self.spatial:
            spatials = self._process_boxes(box_locations,W,H)
            spatials[L:] = 0
            box_locations[L:] = 0
            return L,W,H,box_feats,spatials,box_locations
        return L,W,H,box_feats, box_locations    
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        sent_id = ent['question_id']
        img_id = ent['image_id']
        ans = 0  
        W = ent['width']
        H = ent['height']
        que = ent['question']
        #0  to N boxes
        gtbox = ent['gtbox']
        L_gtboxes = len(gtbox)
        Max_box = 15 #the max number of boxes  in VQD
        if len(gtbox[0]) == 0:
        	gtbox = [[0,0,1,1.0]]*Max_box
        else:
            gtbox = gtbox + [[0,0,1,1.0]]*(Max_box - L_gtboxes)
        
        gtbox = torch.tensor(gtbox).float()
        #boxes from refcoc is in xywh format
        gtboxorig = convert_xywh_x1y1x2y2(gtbox)

        L, W, H ,box_feats,box_coords_6d, box_coordsorig = self._load_image_coco(img_id)        
        box_coords_6d = torch.from_numpy(box_coords_6d)
        
        #boxes in h4files are in x1 y1 x2 y2 format
        iou = getIOU(gtboxorig.unsqueeze(1),torch.from_numpy(box_coordsorig)).squeeze(-1)
        correct = iou>0.5        
        correct = correct.sum(dim=0).clamp(max=1)
        
        _,idxall = torch.max(iou,dim=1)
        #maybe more than one indices so sample for now
        idx = torch.tensor([int(np.random.choice(idxall))])
        idx = torch.tensor([int(idxall[0])])
               
#        print (iou,iou.shape,box_coordsorig,"index",idx)
        gtboxiou = box_coordsorig[idx]
        gtboxiou = torch.from_numpy(gtboxiou)
        
        tokens = tokenize_ques(self.dictionary,que)
        qfeat = torch.from_numpy(tokens).long()
        #tortal number of entries
        N = box_coordsorig.shape[0]
        Lvec = torch.zeros(N).long()
        Lvec[:L] = 1       
        return sent_id,ans,box_feats,box_coordsorig,box_coords_6d.float(),\
                gtboxorig[0].float(),qfeat,Lvec,idx,correct.view(-1)



#%%
if __name__ == "__main__":
    import config   
    ds = 'vqd'
    config.global_config['dictionaryfile'] = config.global_config['dictionaryfile'].format(ds)
    config.global_config['glove'] = config.global_config['glove'].format(ds)      
    dataloader_kwargs = {}
    dataloader_kwargs = {**config.global_config , **config.dataset[ds] }
    dataloader_kwargs['split'] = 'train'
    cd = ReferDataset(**dataloader_kwargs)
    it = iter(cd)
#%%
    data =  next(it)
    print (data)
    sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx,correct = data
      
  