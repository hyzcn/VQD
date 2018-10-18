from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from models.dictionary import Dictionary
from models.language import tokenize_ques
from refer import REFER
import os.path as osp
import json
import copy


class ReferDataset(Dataset):

    def __init__(self,**kwargs):
    
        data_root = kwargs.get('data_root')
        dataset = kwargs.get('dataset')
        splitBy = kwargs.get('splitBy')
        splits = kwargs.get('splits')
        refer = REFER(data_root, dataset, splitBy)
               
        # print stats about the given dataset
        print ('dataset [%s_%s] contains: ' % (dataset, splitBy))
        ref_ids = refer.getRefIds()
        image_ids = refer.getImgIds()
        print ('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))
        
        split_ref_ids = {}
        for split in splits:
            ref_ids = refer.getRefIds(split=split)
            split_ref_ids[split] = ref_ids
            print ('%s refs are in split [%s].' % (len(ref_ids), split))
            
            
        #have to sample various sentences and their tokens from here.
        refs  = split_ref_ids['val']
        self.data = []
        for ref_id in refs:
            ref = refer.Refs[ref_id]
            image_id = ref['image_id']
            sentences = ref.pop('sentences')
            ref.pop('sent_ids')
            entnew = copy.deepcopy(ref)
            anns = refer.imgToAnns[image_id]
            entnew['boxes'] = []
            for box_ann in anns:
                entnew['boxes'].append(box_ann['bbox'])
            for sentence in sentences:
                entnew['sentence'] = sentence
                #entnew['bbox'] = anns['bbox']
                entnew['gtbox'] = refer.refToAnn[ref_id]['bbox']
                self.data.append(entnew)
                

        dictfile = kwargs.get('dictionaryfile').format(dataset)
        self.dictionary = Dictionary.load_from_file(dictfile)    
        if kwargs.get('testrun'):
            self.data = self.data[:20]
            
        self.spatial = True            
        self.image_features_path_coco = kwargs.get('coco_bottomup')
        self.coco_id_to_index =  self.id_to_index(self.image_features_path_coco)  
   

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
        # find the boxes with all co-ordinates 0,0,0,0
        #L = np.where(~box_locations.any(axis=1))[0][0]
                
        if self.spatial:
            spatials = self._process_boxes(box_locations.T,W,H)
            return L,W,H,box_feats.T,spatials
        return L,W,H,box_feats.T, box_locations.T 
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        sent_id = ent['sentence']['sent_id']
        file_name = ent['file_name']
        img_id = ent['image_id']
        ans = ent['category_id']
#        que = ent['question']
        
        #widht = 
        #height = 
        
        #split = 
        
        gtbox = [ent['gtbox']]
        gtbox = torch.tensor(gtbox)
        box_coords = ent['boxes']
        box_coords = torch.tensor(box_coords)
        L, W, H ,imgarr,box_coords = self._load_image_coco(img_id)
#        
#
#        if self.trainembd:
#            tokens = tokenize_ques(self.dictionary,que)
#            qfeat = torch.from_numpy(tokens).long()
#      

        return sent_id,ans,box_coords.float(),gtbox.float()

#%%
if __name__ == "__main__":
    import config
    dataloader_kwargs = {}
    ds = 'refcoco'
    dataloader_kwargs = {**config.global_config , **config.dataset[ds] }
    dataloader_kwargs['dataset'] = ds
    cd = ReferDataset(**dataloader_kwargs)
    it = iter(cd)
    data =  next(it)
    print (data)
    sent_id,ans,box_coords,gtbox = data
      
  