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



from models import Point_GTU , Point_noGTU , Point_RN , Point_noGTU_cb,Point_batchnormcb,Point_tanh
#model names
models = { 
          'GTU': Point_GTU.RN,
          'noGTU': Point_noGTU.RN,
          'noGTUcb': Point_noGTU_cb.RN,
          'tanh': Point_tanh.RN,
          'batchnormcb':Point_batchnormcb.RN,
          'RN': Point_RN.RN,
          } 