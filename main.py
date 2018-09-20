import argparse
import torch
import os
from utils import Logger
import config
from data import CountDataset
#from data_baselines import CountDataset
#from data_vqa import CountDataset
from torch.utils.data import  DataLoader
from train import run
import inspect
from utils import load_checkpoint
from utils import get_current_time
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='HowmanyQA')
    parser.add_argument('--epochs', type=int,help='Number of epochs',default=50)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='RN_OG_embd')
    parser.add_argument('--lr', type=float,default=0.0003,help='Learning rate')
    parser.add_argument('--bs', type=int,default=32,help='Batch size')
    parser.add_argument('--save', help='save folder name',default='extrairlc')
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=str, default=None, help='resume file name')
    parser.add_argument('--test', type=bool, default=False, help='test only')
    parser.add_argument('--testrun', action='store_true', help='test run with few dataset')
    parser.add_argument('--nobaselines', action='store_true',help='does not eval baselines')
    parser.add_argument('--savejson',type=bool,default=True,help='save json in VQA format')
    parser.add_argument('--clip_norm', type=float, default=200.0, help='norm clipping')
    parser.add_argument('--expl', type=str, default='info', help='extra explanation of the method')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    isVQAeval = False
    if args.dsname != 'Ourdb':
        isVQAeval = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    ds = config.dataset[args.dsname]
    N_classes = ds['N_classes']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    loader_kwargs = {'num_workers': 0} if use_cuda else {}
    model = config.models.get(args.model,None)
    if model is None:
        print ("Model name not found valid names are: {} ".format(config.models))
        sys.exit(0)
    model = model(N_classes,trainembd=args.trainembd,**config.global_config)
    model = model.to(device)
    
    savefolder = '_'.join([args.dsname,args.model,args.save])
    logger = Logger(os.path.join(savefolder, 'log.txt'))
    logger.write("==== {} ====".format(get_current_time()))   
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
   
    start_epoch = 0
    if args.resume:
         start_epoch,meta = load_checkpoint(args.resume,model,optimizer)
    else:
        logger.write("== {} ==".format(args.expl))
        logger.write(str(args).replace(',',',\n'))
        #log source code of model being used
        logger.write_silent(inspect.getsource(type(model)))
        logger.write_silent(repr(model))


    dskwargs = { 'trainembd':args.trainembd , 'isnms':args.isnms ,
                'testrun':args.testrun , **config.global_config}
    testds = CountDataset(file = ds['test'],istrain=False,**dskwargs)
    trainds = CountDataset(file = ds['train'],istrain=True,**dskwargs)
    

    test_loader = DataLoader(testds, batch_size=args.bs,
                             shuffle=False, **loader_kwargs)
    train_loader = DataLoader(trainds, batch_size=args.bs,
                         shuffle=True, **loader_kwargs)
    
    run_kwargs = {  **args.__dict__,
                    'start_epoch': start_epoch,
                     'jsonfolder': config.global_config['jsonfolder'],
                     'N_classes': N_classes,
                     'savefolder': savefolder, 
                     'isVQAeval': isVQAeval,
                     'device' : device, 
                     'model' :  model,
                     'train_loader': train_loader,
                     'test_loader': test_loader,
                     'optimizer' : optimizer,
                     'logger':logger,                    
                  }

    run(**run_kwargs)
