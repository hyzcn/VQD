import torch
import os
import config
from utils import load_checkpoint
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import  DataLoader
from opt import parse_args
from data import ReferDataset
import torch.nn as nn
import torch.nn.functional as F

DIR = 'testboxes'
if not os.path.exists(DIR):
    os.mkdir(DIR)

   
def get_image_name_old(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
    return format%(subtype, subtype, image_id)

def retbox(bbox,format='xyxy'):    
    """A utility function to return box coords asvisualizing boxes."""
    if format =='xyxy':
        xmin, ymin, xmax, ymax = bbox
    elif format =='xywh':
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h       
    
    box =  np.array([[xmin, xmax, xmax, xmin, xmin],
                    [ymin, ymin, ymax, ymax, ymin]])
    return box.T

def saveimage(ent,boxes):
    
    image = ent['image_info']['file_name']
    if "train2014" in image:
        image = os.path.join('/home/manoj/train2014',image)
    else:
        image = os.path.join('/home/manoj/val2014',image)
        
    image_id = ent['image_id']
    sent_id = ent['sentence']['sent_id']
    npimg = Image.open(image)      
    plt.figure()
    plt.imshow(npimg)
    ansidx = ent['gtnms']
    
    scores = ent['scores']
    classify = ent['cls']
    clspred = ent['pred']
    for i in range(ent['L']):
       xmin,ymin,xmax,ymax  = boxes[i]
       x =[xmin,ymin,xmax,ymax]
       rect = retbox(x)
       alpha = np.abs(scores[i])/ np.max(np.abs(scores))
       #alpha = abs(scores[i])
       if i == ansidx:
           plt.plot(rect[:,0],rect[:,1],'g',linewidth=3.0)
           loc = (xmin,0.5*(ymin+ymax))
           plt.text(*loc,"{:.2f}, {:d}".format(scores[i],classify[i]),color='g', fontsize=8)

       if i == clspred:
           plt.plot(rect[:,0],rect[:,1],'r-.',linewidth=2.0)
           loc = (0.5*(xmin+xmax),0.5*(ymin+ymax))
           plt.text(*loc,"{:.2f}, {:d}".format(scores[i],classify[i]),color='r', fontsize=8)

       else:
           plt.plot(rect[:,0],rect[:,1],'y',alpha = alpha,linewidth=1.0)
           plt.text(xmin,ymin,"{:.2f}[{:d}]".format(scores[i],classify[i]),color='c', fontsize=7,alpha = alpha)
       
       
    cocogt = retbox(ent['gtbox'],format='xywh')
    plt.plot(cocogt[:,0],cocogt[:,1],'k',linewidth=3.0)          
           
    question = ent['sentence']['raw']
    imglast = image.split("/")[-1]
    plt.title("Pred index: {} .. G = GT, K = COCOGT , R = pred".format(clspred))
    plt.xlabel("{}".format(question))
    path = os.path.join(DIR,"ann_{}_{}".format(sent_id,imglast))
    plt.savefig(path,dpi=150)
    plt.close()



if __name__ == '__main__':
    
    args = parse_args()
    opt = vars(args)
       

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")        
    loader_kwargs = {'num_workers': 0} if use_cuda else {}
    eval_split = args.evalsplit 
    N_classes = 20 # xxx change this
    model = config.models.get(args.model,None)
    if model is None:
        print ("Model name not found valid names are: {} ".format(config.models))
        sys.exit(0)
        
    config.global_config['dictionaryfile'] = config.global_config['dictionaryfile'].format(args.dataset)
    config.global_config['glove'] = config.global_config['glove'].format(args.dataset)        
    model = model(N_classes,**config.global_config)
    model = model.to(device)
    
    savefolder = '_'.join([args.dataset,args.model,args.save]) 
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    start_epoch = 0
    if args.resume:
         start_epoch,meta = load_checkpoint(args.resume,model,optimizer)
        

    dskwargs = { 'testrun':args.testrun ,
                **config.global_config, **config.dataset[args.dataset]}
    
    testdset = ReferDataset(split = eval_split ,istrain=False,**dskwargs)
    test_loader = DataLoader(testdset, batch_size= 1 ,
                                 shuffle=False, **loader_kwargs)
        
    testds = test_loader.dataset.data
    with torch.set_grad_enabled(False):
        for i,data in enumerate(test_loader):
            sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx = data
            ent = testds[i]

    
            print (ent)      
              
    
            #normalize the box feats
            box_feats = F.normalize(box_feats,p=2,dim=-1)
            box_feats = box_feats.to(device)
            box_coords_6d = box_coords_6d.to(device)
            q_feats = qfeat.to(device)
            idx = idx.long()
            optimizer.zero_grad()
            
            B = box_feats.shape[0]
            Nbox = box_feats.shape[1]
            
            net_kwargs = { 'box_feats':box_feats,
                           'q_feats':q_feats,
                           'box_coords':box_coords_6d}
    
            scores,logits = model(**net_kwargs)  
            logits = logits.view(B*Nbox,-1)
            scores = scores.squeeze()                                   
            maxscore,clspred = torch.max(scores,-1)

            ent['L'] = L.item()
            ent['gtnms'] = int(idx.item())            
            ent['scores'] = scores.tolist()
            ent['cls'] =  torch.argmax(logits,dim=1).tolist()
            ent['pred'] =  int(clspred)
            print (scores,ent['cls'])              
            print ("correct box index: {}".format(idx))            
            print ("pred box index: {}".format(clspred))       
            
                   
            saveimage(ent,box_coordsorig.tolist()[0])       
            feedback = input("Continue [N/n]?: ")
            if feedback in ['N','n']:
                print ("Done....")
                sys.exit(0)
           


