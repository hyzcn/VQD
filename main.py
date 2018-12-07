import torch
import os
import config
import sys
from torch.utils.data import  DataLoader
import inspect
from utils import load_checkpoint
from utils import get_current_time
from utils import Logger
from opt import parse_args
from data_vqd import ReferDataset
from train import run

if __name__ == '__main__':
    
    args = parse_args()
    opt = vars(args)
       

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")        
    loader_kwargs = {'num_workers': args.workers} if use_cuda else {}
    eval_split = args.evalsplit 
    if eval_split not in config.dataset[args.dataset]['splits']:
        print ("{} split not found in [{}]!!".format(eval_split,args.dataset))
        sys.exit(0)
    
    N_classes = 20 # xxx change this
    model = config.models.get(args.model,None)
    if model is None:
        print ("Model name not found !!\nvalid names are: {} ".format(config.models))
        sys.exit(0)
        
    config.global_config['dictionaryfile'] = config.global_config['dictionaryfile'].format(args.dataset)
    config.global_config['glove'] = config.global_config['glove'].format(args.dataset)        
    model = model(N_classes,**config.global_config)
    model = model.to(device)
    
    savefolder = '_'.join([args.dataset,args.model,args.save])
    logger = Logger(os.path.join(savefolder, 'log.txt'))
    logger.write("==== {} ====".format(get_current_time()))   
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    start_epoch = 0
    if args.load:
         start_epoch,meta = load_checkpoint(args.load,model,optimizer)
    else:
        logger.write("== {} ==".format(args.expl))
        logger.write(str(args).replace(',',',\n'))
        #log source code of model being used
        logger.write_silent(inspect.getsource(type(model)))
        logger.write_silent(repr(model))
        

    dskwargs = { 'testrun':args.testrun ,
                **config.global_config, **config.dataset[args.dataset]}
    
   
    trainds = ReferDataset(split = 'train' ,istrain=True,**dskwargs)
    train_loader = DataLoader(trainds, batch_size=args.batch_size,
                         shuffle=True, **loader_kwargs)
    
    testds = ReferDataset(split = eval_split ,istrain=False,**dskwargs)
    test_loader = DataLoader(testds, batch_size=args.batch_size,
                                 shuffle=False, **loader_kwargs)

    run_kwargs = {   **vars(args),
                     **config.global_config,
                     'start_epoch': start_epoch,
                     'N_classes': N_classes,
                     'savefolder': savefolder, 
                     'device' : device, 
                     'model' :  model,
                     'train_loader': train_loader,
                     'test_loader': test_loader,
                     'optimizer' : optimizer,
                     'logger':logger,                    
                  }

    run(**run_kwargs)
