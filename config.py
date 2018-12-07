from collections import defaultdict
dataset = defaultdict(list)

#global config
global_config = {}
global_config['data_root'] = 'data/'
global_config['coco_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152.h5'
global_config['genome_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152_genome.h5'
global_config['coco_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-trainval.h5'
global_config['genome_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome_ourdb/genome-trainval.h5'
global_config['refcoco_frcnn'] = '/home/manoj/reffeats/'

global_config['vqd_detfeats'] = '/home/manoj/Music/faster-rcnn.pytorch/output/res101/coco_2014_val/faster_rcnn_10/coco_{}2014_det_feats.h5'



#dictionary
global_config['dictionaryfile'] = 'embedding/{}/dictionary.pickle'
global_config['glove'] = 'embedding/{}/glove6b_init_300d.npy'

#dataset configs
name= 'refcoco'
dataset[name] = {}
dataset[name]['dataset'] = name
dataset[name]['splitBy'] = 'unc'
dataset[name]['splits'] = ['val','testA','testB']

name= 'refcoco+'
dataset[name] = {}
dataset[name]['dataset'] = name
dataset[name]['splitBy'] = 'unc'
dataset[name]['splits'] = ['val','testA','testB']

name= 'refcocog'
dataset[name] = {}
dataset[name]['dataset'] = name
dataset[name]['splitBy'] = 'umd'
dataset[name]['splits'] = ['val','test']

name= 'vqd'
dataset[name] = {}
dataset[name]['dataset'] = name
dataset[name]['splitBy'] = ''
dataset[name]['splits'] = ['val']


name= 'vqd1'
dataset[name] = {}
dataset[name]['dataset'] = name
dataset[name]['splitBy'] = ''
dataset[name]['splits'] = ['val']



#model names
from models import Point_GTU , Point_noGTU , Point_RN ,\
                    Point_noGTU_cb,Point_batchnormcb,Point_tanh,\
                    Point_boxonly,Point_final,Point_qbox

models = { 
          'GTU': Point_GTU.RN,
          'noGTU': Point_noGTU.RN,
          'noGTUcb': Point_noGTU_cb.RN,
          'tanh': Point_tanh.RN,
          'batchnormcb':Point_batchnormcb.RN,
          'boxonly':Point_boxonly.RN,
          'qbox':Point_qbox.RN,
          'RN': Point_RN.RN,
          'final':Point_final.RN,
          } 


#%%

coco_categories = [{'id': 1, 'name': 'person', 'supercategory': 'person'},
 {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'},
 {'id': 3, 'name': 'car', 'supercategory': 'vehicle'},
 {'id': 4, 'name': 'motorcycle', 'supercategory': 'vehicle'},
 {'id': 5, 'name': 'airplane', 'supercategory': 'vehicle'},
 {'id': 6, 'name': 'bus', 'supercategory': 'vehicle'},
 {'id': 7, 'name': 'train', 'supercategory': 'vehicle'},
 {'id': 8, 'name': 'truck', 'supercategory': 'vehicle'},
 {'id': 9, 'name': 'boat', 'supercategory': 'vehicle'},
 {'id': 10, 'name': 'traffic light', 'supercategory': 'outdoor'},
 {'id': 11, 'name': 'fire hydrant', 'supercategory': 'outdoor'},
 {'id': 13, 'name': 'stop sign', 'supercategory': 'outdoor'},
 {'id': 14, 'name': 'parking meter', 'supercategory': 'outdoor'},
 {'id': 15, 'name': 'bench', 'supercategory': 'outdoor'},
 {'id': 16, 'name': 'bird', 'supercategory': 'animal'},
 {'id': 17, 'name': 'cat', 'supercategory': 'animal'},
 {'id': 18, 'name': 'dog', 'supercategory': 'animal'},
 {'id': 19, 'name': 'horse', 'supercategory': 'animal'},
 {'id': 20, 'name': 'sheep', 'supercategory': 'animal'},
 {'id': 21, 'name': 'cow', 'supercategory': 'animal'},
 {'id': 22, 'name': 'elephant', 'supercategory': 'animal'},
 {'id': 23, 'name': 'bear', 'supercategory': 'animal'},
 {'id': 24, 'name': 'zebra', 'supercategory': 'animal'},
 {'id': 25, 'name': 'giraffe', 'supercategory': 'animal'},
 {'id': 27, 'name': 'backpack', 'supercategory': 'accessory'},
 {'id': 28, 'name': 'umbrella', 'supercategory': 'accessory'},
 {'id': 31, 'name': 'handbag', 'supercategory': 'accessory'},
 {'id': 32, 'name': 'tie', 'supercategory': 'accessory'},
 {'id': 33, 'name': 'suitcase', 'supercategory': 'accessory'},
 {'id': 34, 'name': 'frisbee', 'supercategory': 'sports'},
 {'id': 35, 'name': 'skis', 'supercategory': 'sports'},
 {'id': 36, 'name': 'snowboard', 'supercategory': 'sports'},
 {'id': 37, 'name': 'sports ball', 'supercategory': 'sports'},
 {'id': 38, 'name': 'kite', 'supercategory': 'sports'},
 {'id': 39, 'name': 'baseball bat', 'supercategory': 'sports'},
 {'id': 40, 'name': 'baseball glove', 'supercategory': 'sports'},
 {'id': 41, 'name': 'skateboard', 'supercategory': 'sports'},
 {'id': 42, 'name': 'surfboard', 'supercategory': 'sports'},
 {'id': 43, 'name': 'tennis racket', 'supercategory': 'sports'},
 {'id': 44, 'name': 'bottle', 'supercategory': 'kitchen'},
 {'id': 46, 'name': 'wine glass', 'supercategory': 'kitchen'},
 {'id': 47, 'name': 'cup', 'supercategory': 'kitchen'},
 {'id': 48, 'name': 'fork', 'supercategory': 'kitchen'},
 {'id': 49, 'name': 'knife', 'supercategory': 'kitchen'},
 {'id': 50, 'name': 'spoon', 'supercategory': 'kitchen'},
 {'id': 51, 'name': 'bowl', 'supercategory': 'kitchen'},
 {'id': 52, 'name': 'banana', 'supercategory': 'food'},
 {'id': 53, 'name': 'apple', 'supercategory': 'food'},
 {'id': 54, 'name': 'sandwich', 'supercategory': 'food'},
 {'id': 55, 'name': 'orange', 'supercategory': 'food'},
 {'id': 56, 'name': 'broccoli', 'supercategory': 'food'},
 {'id': 57, 'name': 'carrot', 'supercategory': 'food'},
 {'id': 58, 'name': 'hot dog', 'supercategory': 'food'},
 {'id': 59, 'name': 'pizza', 'supercategory': 'food'},
 {'id': 60, 'name': 'donut', 'supercategory': 'food'},
 {'id': 61, 'name': 'cake', 'supercategory': 'food'},
 {'id': 62, 'name': 'chair', 'supercategory': 'furniture'},
 {'id': 63, 'name': 'couch', 'supercategory': 'furniture'},
 {'id': 64, 'name': 'potted plant', 'supercategory': 'furniture'},
 {'id': 65, 'name': 'bed', 'supercategory': 'furniture'},
 {'id': 67, 'name': 'dining table', 'supercategory': 'furniture'},
 {'id': 70, 'name': 'toilet', 'supercategory': 'furniture'},
 {'id': 72, 'name': 'tv', 'supercategory': 'electronic'},
 {'id': 73, 'name': 'laptop', 'supercategory': 'electronic'},
 {'id': 74, 'name': 'mouse', 'supercategory': 'electronic'},
 {'id': 75, 'name': 'remote', 'supercategory': 'electronic'},
 {'id': 76, 'name': 'keyboard', 'supercategory': 'electronic'},
 {'id': 77, 'name': 'cell phone', 'supercategory': 'electronic'},
 {'id': 78, 'name': 'microwave', 'supercategory': 'appliance'},
 {'id': 79, 'name': 'oven', 'supercategory': 'appliance'},
 {'id': 80, 'name': 'toaster', 'supercategory': 'appliance'},
 {'id': 81, 'name': 'sink', 'supercategory': 'appliance'},
 {'id': 82, 'name': 'refrigerator', 'supercategory': 'appliance'},
 {'id': 84, 'name': 'book', 'supercategory': 'indoor'},
 {'id': 85, 'name': 'clock', 'supercategory': 'indoor'},
 {'id': 86, 'name': 'vase', 'supercategory': 'indoor'},
 {'id': 87, 'name': 'scissors', 'supercategory': 'indoor'},
 {'id': 88, 'name': 'teddy bear', 'supercategory': 'indoor'},
 {'id': 89, 'name': 'hair drier', 'supercategory': 'indoor'},
 {'id': 90, 'name': 'toothbrush', 'supercategory': 'indoor'}]


coco_classes = [ ent['id'] for ent in coco_categories]
cocoid2label = { ent['id']:ent['name'] for ent in coco_categories}
cocolabel2id = { ent['name']:ent['id'] for ent in coco_categories}
