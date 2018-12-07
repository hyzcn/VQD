import torch
import torch.nn as nn
import time
from utils import AverageMeter
from models.callbacks import EarlyStopping
import torch.nn.functional as F
from utils import save_checkpoint
import utils
import os
import json
import numpy as np
import eval_extra
#from CLR import CyclicLR
from utils import adjust_learning_rate
from calculate_mean_ap import get_avg_precision_at_iou



#%%


#a = torch.tensor([1,2,3.0])
#b = torch.tensor([0])
#print (mmloss(a,b))

def main(**kwargs):
    
    device = kwargs.get('device') 
    net = kwargs.get('model')                     
    optimizer = kwargs.get('optimizer')
    epoch = kwargs.get('epoch')
    istrain = kwargs.get('istrain')
  
    start_time = time.time()
    pred = {}
    #idxs = []
    loss_meter = AverageMeter()
    loss_meter.reset()
    
    Nprint = 100
    if istrain:
        net.train()
        loader = kwargs.get('train_loader')
    else:
        loader = kwargs.get('test_loader')
        net.eval()

    clslossfn = nn.CrossEntropyLoss(reduction='none')
    #mmloss = nn.MultiMarginLoss()
    mmloss = nn.CrossEntropyLoss()
    
    with torch.set_grad_enabled(istrain):
        for imain, data in enumerate(loader):
            sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx,correct = data            
            #idxs.extend(sent_id.tolist())        
            #true.extend(gtbox.tolist())
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
                           'box_coords':box_coords_6d,
                           'index':L}

            scores,logits = net(**net_kwargs)            
            logits = logits.view(B*Nbox,-1)
            
            #idx_expand = correct.view(-1)
               
            idx_expand_cls = correct
            #idx_expand_cls = torch.mul(correct.long(),ans.unsqueeze(1))
            idx_expand_cls = idx_expand_cls.view(-1).to(device)
            
                
            loss_cc_allbox = clslossfn(logits.view(B*Nbox,-1), idx_expand_cls.long())            
            loss_cc_allbox = loss_cc_allbox[ L.view(-1) == 1]
            len_true_box = L.sum().item()
            loss_cc = torch.sum( loss_cc_allbox)/len_true_box
            
            
            scores = scores.squeeze()    
            # TODO: will this affect the gradients in any way
            #should we directly add to data             
            L = L.to(device)           
            scores_idx = torch.cat( (torch.tensor(range(0,B)).unsqueeze(1).long(),idx.long()),dim=1)
            scores_det = scores.detach()
            gt_scores = scores_det[scores_idx[:,0],scores_idx[:,1]].unsqueeze(1)
            ignore_index_scores = torch.mul(gt_scores,(L==0).float())            
            scores_det  =  ignore_index_scores + torch.mul(scores_det,L.float())

            #Losses
            loss_margin = mmloss(scores,idx.to(device).squeeze()) 
            loss = loss_cc + loss_margin     
            
            #get scores ans get max scores
            _,clspred = torch.max(scores,-1)           
    
            scores_lst = scores.tolist()
            clsp = clspred.tolist()
            for i,qid in enumerate(sent_id.tolist()):
                dent = {}
                Nbox = clsp[i]
                allboxes = box_coordsorig[i].tolist()
                allscores = scores_lst[i]              
                Nbox = clspred[i].item()
                dent['boxes'] =  [allboxes[Nbox]]
                dent['scores'] = [0.99]            
                #ent['boxes'] =  allboxes[:Nbox]
                #ent['scores'] = [:Nbox]
                pred[qid] = dent
            
       
            loss_meter.update(loss.item())   
            if istrain:
                #scheduler.step()
                loss.backward()
                #gradient clipping
                if kwargs.get('clip_norm'):
                    nn.utils.clip_grad_norm_(net.parameters(), kwargs.get('clip_norm'))
                optimizer.step()
    
            if imain == 0 and epoch == 0 and istrain:
                    print ("Starting loss: {:.4f}".format(loss.item()))
                
    
            if imain % Nprint == Nprint-1:
                infostr = "Epoch [{}]:Iter [{}]/[{}] Loss: {:.4f} Time: {:2.2f} s"
                printinfo = infostr.format(epoch , i, len(loader),
                                           loss_meter.avg,time.time() - start_time)
    
                print (printinfo)
        
        print("Completed in: {:2.2f} s".format(time.time() - start_time))
        ent = {}
        ent['pred'] = pred
        ent['loss'] = loss_meter.avg
        return ent


def getGTboxes(data,**kwargs):
    gt = {}
    #print (data[10],len(data))
    if 'vqd' in kwargs.get('dataset'):
        for entry in data:
            qid = entry['question_id']
            gtbox = entry['gtbox']
            if len(gtbox[0]) == 0:
                    gtbox =[[0,0,0.0,0.0]]            
            else:
                xywh = np.array(gtbox)
                gtbox = utils.xywh_to_xyxy(xywh).tolist()     
            gt[qid] = gtbox
    else:             
        for entry in data:
            qid = entry['sentence']['sent_id']
            xywh = np.array([entry['gtbox']])
            gtbox = utils.xywh_to_xyxy(xywh)     
            gt[qid] = gtbox.tolist()

    return gt


def testonly(**kwargs):
    epoch = kwargs.get('start_epoch')   
    testloader = kwargs.get('test_loader')
    kwargs['epoch'] = epoch
    start_time = time.time()
    test =  main(istrain=False,**kwargs)
    total_time  = time.time() - start_time

    print('Epoch {} Time {:2.2f} s ------'.format(epoch,total_time))
    print('\tTest Loss: {:.4f}'.format(test['loss']))
 
          
    gt = {}
    gt['test'] = getGTboxes(testloader.dataset.data,**kwargs)    
    datatest = get_avg_precision_at_iou(gt['test'], test['pred'], iou_thr= 0.5)
    print(' Test avg precision: {:.4f}'.format(datatest['avg_prec']))



def run(**kwargs):

    savefolder = kwargs.get('savefolder')
    logger = kwargs.get('logger')
    epochs = kwargs.get('epochs')
    savemodel = kwargs.get('savemodel')
    #there are many test loaders
    start_epoch = kwargs.get('start_epoch')
    trainloader = kwargs.get('train_loader')
    testloader = kwargs.get('test_loader')
    #test only mode        
    if kwargs.get('test') == True:
        testonly(**kwargs)
        return 0
        

    early_stop = EarlyStopping(monitor='loss',patience=8)   
    Modelsavefreq = 1
           
    gt = {}
    gt['test'] = getGTboxes(testloader.dataset.data,**kwargs)
    gt['train'] = getGTboxes(trainloader.dataset.data,**kwargs)

    for epoch in range(start_epoch,epochs):

        kwargs['epoch'] = epoch
        start_time = time.time()
        train = main(istrain=True,**kwargs)
        test =  main(istrain=False,**kwargs)
        total_time  = time.time() - start_time

        logger.write('Epoch {} Time {:2.2f} s ------'.format(epoch,total_time))
        logger.write('\tTrain Loss: {:.4f}'.format(train['loss']))
        logger.write('\tTest Loss: {:.4f}'.format(test['loss']))
        
        datatrain = get_avg_precision_at_iou(gt['train'], train['pred'], iou_thr= 0.5)
        logger.write('Train mAP: {:.4f}'.format(datatrain['avg_prec']))
        
        datatest = get_avg_precision_at_iou(gt['test'], test['pred'], iou_thr= 0.5)
        logger.write('Test mAP: {:.4f}'.format(datatest['avg_prec']))
        
        #log extra information in a logger dict
        logger.append('train loss',train['loss'])
        logger.append('test loss',test['loss'])
        logger.append('train acc',100*datatrain['avg_prec'])
        logger.append('test acc',100*datatest['avg_prec'])        
        

        if kwargs.get('savejson'):
            path = os.path.join(savefolder, 'test{}.json'.format(epoch))
            json.dump(test['pred'],open(path,'w'))
            
        is_best = False
        if epoch % Modelsavefreq == 0:
            if savemodel or is_best:
                print ('Saving model ....')
                tbs = {'epoch': epoch,
                    'state_dict': kwargs.get('model').state_dict(),
                    'optimizer' : kwargs.get('optimizer').state_dict()}  
                save_checkpoint(savefolder,tbs,is_best)

        logger.dump_info()
                
        early_stop.on_epoch_end(epoch,logs=test)
        if early_stop.stop_training:
            lr =  kwargs.get('optimizer').param_groups[0]['lr']
            adjust_learning_rate(kwargs.get('optimizer'), lr* 0.8)
            logger.write("New Learning rate: {} ".format(lr))
            early_stop.reset()
            #break
            
    logger.write('Finished Training')


