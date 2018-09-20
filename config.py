from collections import defaultdict
from models.baseline import Qmodel,Imodel,QImodel
from models import Point
dataset = defaultdict(list)

#global config
global_config = {}
global_config['jsonfolder'] ='/home/manoj/mutan'
global_config['coco_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152.h5'
global_config['genome_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152_genome.h5'
global_config['coco_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-trainval.h5'
global_config['cocotest_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-test.h5'
global_config['genome_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome_ourdb/genome-trainval.h5'

#dictionary
global_config['dictionaryfile'] = 'data/dictionary.pickle'
global_config['glove'] = 'data/glove6b_init_300d.npy'

#dataset configs
name= 'Ourdb'
dataset[name] = {}
dataset[name]['N_classes'] = 16
dataset[name]['test'] = '/home/manoj/Downloads/counting/data/test_data_kushalformat_fixed.pkl'
dataset[name]['train'] = '/home/manoj/Downloads/counting/data/train_data_kushalformat_fixed.pkl'

#model names
models = { 'Q':Qmodel, 'I': Imodel, 'QI': QImodel,
          'mainmodel': Point         
          } 
