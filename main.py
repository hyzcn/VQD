import torch
import os
import config
from torch.utils.data import  DataLoader
import inspect
from utils import load_checkpoint
from utils import get_current_time
from utils import Logger
from opt import parse_args
import os.path as osp
from data import ReferDataset
#from train import run

if __name__ == '__main__':
    
    args = parse_args()
    opt = vars(args)
       

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set up loader
    data_json = osp.join('cache/prepro', args.dataset, 'data.json')
    checkpoint_dir = osp.join('cache/prepro', args.dataset)
    if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")        
    loader_kwargs = {'num_workers': 0} if use_cuda else {}
    
##    model = config.models.get(args.model,None)
##    if model is None:
##        print ("Model name not found valid names are: {} ".format(config.models))
##        sys.exit(0)
##    model = model(N_classes,trainembd=args.trainembd,**config.global_config)
##    model = model.to(device)
##    
#    savefolder = '_'.join([opt['dataset_splitBy'],args.model,args.save])
#    logger = Logger(os.path.join(savefolder, 'log.txt'))
#    logger.write("==== {} ====".format(get_current_time()))   
#    print (model)
#
#    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
#
#    start_epoch = 0
#    if args.resume:
#         start_epoch,meta = load_checkpoint(args.resume,model,optimizer)
#    else:
#        logger.write("== {} ==".format(args.expl))
#        logger.write(str(args).replace(',',',\n'))
#        #log source code of model being used
#        logger.write_silent(inspect.getsource(type(model)))
#        logger.write_silent(repr(model))
        
        
#
#    dskwargs = { 'trainembd':args.trainembd , 'isnms':args.isnms ,
#                'testrun':args.testrun , **config.global_config}
#    testds = CountDataset(file = ds['test'],istrain=False,**dskwargs)
#    trainds = CountDataset(file = ds['train'],istrain=True,**dskwargs)
#    
#
#    test_loader = DataLoader(testds, batch_size=args.bs,
#                             shuffle=False, **loader_kwargs)
#    train_loader = DataLoader(trainds, batch_size=args.bs,
#                         shuffle=True, **loader_kwargs)
#    
#    run_kwargs = {  **args.__dict__,
#                    'start_epoch': start_epoch,
#                     'jsonfolder': config.global_config['jsonfolder'],
#                     'N_classes': N_classes,
#                     'savefolder': savefolder, 
#                     'isVQAeval': isVQAeval,
#                     'device' : device, 
#                     'model' :  model,
#                     'train_loader': train_loader,
#                     'test_loader': test_loader,
#                     'optimizer' : optimizer,
#                     'logger':logger,                    
#                  }
#
#    run(**run_kwargs)
