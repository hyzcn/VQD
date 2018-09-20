from torch.utils.data import Dataset
import numpy as np
import torch
from models.language import getglove
import pickle
import h5py
from models.dictionary import Dictionary
from models.lang_new import tokenize_ques
from nms_expt import non_max_suppression_fast

class CountDataset(Dataset):

    def __init__(self,**kwargs):

        
        file = kwargs.get('file')
        
        self.isnms = kwargs.get('isnms')
        self.trainembd = kwargs.get('trainembd')
        
        #6 postion encoded vectors as used by irls
        self.spatial = True
        
        with open(file,'rb') as f:
            self.data = pickle.load(f)
                                 
        if self.trainembd:    
            self.dictionary = Dictionary.load_from_file(kwargs.get('dictionaryfile'))
        
        if kwargs.get('testrun'):
            self.data = self.data[:32]
             
        self.pool_features_path_coco = kwargs.get('coco_pool_features')
        self.pool_features_path_genome = kwargs.get('genome_pool_features')
        self.poolcoco_id_to_index =  self._poolcreate_coco_id_to_index(self.pool_features_path_coco)
        self.poolcoco_id_to_index_gen =  self._poolcreate_coco_id_to_index(self.pool_features_path_genome)        

        self.image_features_path_coco = kwargs.get('coco_bottomup')
        self.coco_id_to_index =  self.id_to_index(self.image_features_path_coco)  
        self.image_features_path_genome = kwargs.get('genome_bottomup')
        self.genome_id_to_index =  self.id_to_index(self.image_features_path_genome)        

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
        
        if self.isnms:
            keep  =  non_max_suppression_fast(box_locations.T,0.7)
            L = len(keep)
            box_f = box_feats.T[keep]
            box_f = np.concatenate([box_f,np.zeros((100-L,2048))])
            box_l = box_locations.T[keep]
            box_l = np.concatenate([box_l,np.zeros((100-L,4))])
            return L,W,H,box_f,box_l  
        
        if self.spatial:
            spatials = self._process_boxes(box_locations.T,W,H)
            return L,W,H,box_feats.T,spatials
        return L,W,H,box_feats.T, box_locations.T 
 

    def _load_image_genome(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file_genome'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file_genome = h5py.File(self.image_features_path_genome, 'r')
           
        image_id = int(str(image_id)[1:])
        index = self.genome_id_to_index[image_id]
        L = self.features_file_genome['num_boxes'][index]
        W = self.features_file_genome['widths'][index]
        H = self.features_file_genome['heights'][index]
        box_feats = self.features_file_genome['features'][index]
        box_locations = self.features_file_genome['boxes'][index]
        # find the boxes with all co-ordinates 0,0,0,0
        #L = np.where(~box_locations.any(axis=1))[0][0]
        
        if self.isnms:
            keep  =  non_max_suppression_fast(box_locations.T,0.7)
            L = len(keep)
            box_f = box_feats.T[keep]
            box_f = np.concatenate([box_f,np.zeros((100-L,2048))])
            box_l = box_locations.T[keep]
            box_l = np.concatenate([box_l,np.zeros((100-L,4))])
            return L,W,H,box_f,box_l  
        if self.spatial:
            spatials = self._process_boxes(box_locations.T,W,H)
            return L,W,H,box_feats.T,spatials
        return L,W,H,box_feats.T, box_locations.T 
        
        
    def _poolcreate_coco_id_to_index(self , path):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(path, 'r') as features_file:
            coco_ids = features_file['filenames'][()]
        coco_id_to_index = {name: i for i, name in enumerate(coco_ids)}
        return coco_id_to_index        
    
   

          
    def _load_pool_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'pool_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.pool_file = h5py.File(self.pool_features_path_coco, 'r')
        if not hasattr(self, 'pool_file_gen'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.pool_file_gen = h5py.File(self.pool_features_path_genome, 'r')          
            
        index = self.poolcoco_id_to_index.get(image_id,None)
        if index is not None:
            whole = self.pool_file['pool5'][index]        
            unpooled = self.pool_file['res5c'][index]
            return torch.from_numpy(whole).float(), torch.from_numpy(unpooled).float()  
        else:
            index = self.poolcoco_id_to_index_gen.get(image_id,None)
            whole = self.pool_file_gen['pool5'][index]        
            unpooled = self.pool_file_gen['res5c'][index]
            return torch.from_numpy(whole).float(), torch.from_numpy(unpooled).float()    

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        qid = ent['question_id']
        img_name = ent['image']
        img_id = ent['image_id']
        ans = ent.get('multiple_choice_answer',None)
        if ans is None:
            ans = ent['answer']
        que = ent['question']
       
        lasttwo = '/'.join(img_name.split("/")[-2:])
        lasttwo +=".pkl"
        lastone = lasttwo.split("/")[-1]
        wholefeat,pooled = self._load_pool_image(lasttwo[:-4])

        #wholefeat = pooled = 0
        
#        pk = pickle.load(open(os.path.join("/home/manoj/448feats/feats",lastone),"rb"))
#        L =  len(pk) - 1 # lenght of entries in pickle file

        if 'VG' in img_name:
            L, W, H ,imgarr,box_coords = self._load_image_genome(img_id)
        else:
            L, W, H ,imgarr,box_coords = self._load_image_coco(img_id)
        

        if self.trainembd:
            tokens = tokenize_ques(self.dictionary,que)
            qfeat = torch.from_numpy(tokens).long()
        else:
            qfeat = getglove(que)
            qfeat = torch.from_numpy(qfeat)

        imgarr = torch.from_numpy(imgarr)
        box_coords = torch.from_numpy(box_coords)
        if not self.spatial:
            scale = torch.tensor([W,H,W,H])
            box_coords = box_coords / scale   
        return qid,wholefeat,pooled,imgarr.float(),np.float32(ans),qfeat,box_coords.float(),L

