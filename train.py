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
    true = []
    pred = []
    idxs = []
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
    mmloss = nn.MultiMarginLoss()
    
    with torch.set_grad_enabled(istrain):
        for i, data in enumerate(loader):
            sent_id,ans,box_feats,box_coordsorig,box_coords_6d,gtbox,qfeat,L,idx,correct = data            
            idxs.extend(sent_id.tolist())        
            true.extend(gtbox.tolist())
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
            
            idx_expand = correct.view(-1)
            idx_expand = idx_expand.to(device)                   
            loss_cc_allbox = clslossfn(logits.view(B*Nbox,-1), idx_expand.long())            
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

            
            loss_margin = mmloss(scores,idx.to(device).squeeze()) 
            loss = loss_cc + loss_margin           
            _,clspred = torch.max(scores,-1)
            
            
            iipred = torch.cat( (torch.tensor(range(0,B)).unsqueeze(1).long(),clspred.cpu().unsqueeze(1).long()),dim=1)
            predbox = box_coordsorig[iipred[:,0],iipred[:,1]]
            pred.extend(predbox.tolist())
        
            loss_meter.update(loss.item())   
            if istrain:
                #scheduler.step()
                loss.backward()
                #gradient clipping
                if kwargs.get('clip_norm'):
                    nn.utils.clip_grad_norm_(net.parameters(), kwargs.get('clip_norm'))
                optimizer.step()
    
            if i == 0 and epoch == 0 and istrain:
                print ("Starting loss: {:.4f}".format(loss.item()))
    
    
            if i % Nprint == Nprint-1:
                infostr = "Epoch [{}]:Iter [{}]/[{}] Loss: {:.4f} Time: {:2.2f} s"
                printinfo = infostr.format(epoch , i, len(loader),
                                           loss_meter.avg,time.time() - start_time)
    
                print (printinfo)
        
        print("Completed in: {:2.2f} s".format(time.time() - start_time))
        ent = {}
        ent['true'] = true
        ent['pred'] = pred
        ent['loss'] = loss_meter.avg
        ent['sent_ids'] = idxs
        return ent



def run(**kwargs):

    savefolder = kwargs.get('savefolder')
    logger = kwargs.get('logger')
    epochs = kwargs.get('epochs')
    #there are many test loaders
    start_epoch = kwargs.get('start_epoch')
    eval_baselines = kwargs.get('nobaselines') == False

    if start_epoch == 0 and eval_baselines: # if not resuming
        pass
        #eval_extra.main(**kwargs)
        

    early_stop = EarlyStopping(monitor='loss',patience=8)   
    Modelsavefreq = 1

    for epoch in range(start_epoch,epochs):

        kwargs['epoch'] = epoch
        start_time = time.time()
        train = main(istrain=True,**kwargs)
        test =  main(istrain=False,**kwargs)
        total_time  = time.time() - start_time

        logger.write('Epoch {} Time {:2.2f} s ------'.format(epoch,total_time))
        logger.write('\tTrain Loss: {:.4f}'.format(train['loss']))
        logger.write('\tTest Loss: {:.4f}'.format(test['loss']))
 
        traingt = torch.tensor(train['true'])
        trainpred = torch.tensor(train['pred'])
        trainacc = eval_extra.getaccuracy(traingt,trainpred)
        logger.write("\tTrain Precision@1/Top 1 precision or Accuracy {:.2f}%".format(trainacc))
              
        predictions = dict(zip(test['sent_ids'] , test['pred']))           
        testgt = torch.tensor(test['true'])
        testpred = torch.tensor(test['pred'])
        testacc = eval_extra.getaccuracy(testgt,testpred)
        logger.write("\tTest Precision@1/Top 1 precision or Accuracy {:.2f}%".format(testacc))
        
        #log extra information in a logger dict
        logger.append('train loss',train['loss'])
        logger.append('test loss',test['loss'])
        logger.append('train acc',trainacc)
        logger.append('test acc',testacc)        
        

        if kwargs.get('savejson'):
            js = []
            for qid in predictions:
                ent = {}
                ent['sent_id'] = int(qid)
                ent['bbox'] = predictions[qid]
                js.append(ent)
            path = os.path.join(savefolder, 'test{}.json'.format(epoch))
            json.dump(js,open(path,'w'))
            
        is_best = False
        if epoch % Modelsavefreq == 0:
            print ('Saving model ....')
            tbs = {
                'epoch': epoch,
                'state_dict': kwargs.get('model').state_dict(),
                'true':test['true'],
                'pred':test['pred'],
                'sent_ids': test['sent_ids'],
                'optimizer' : kwargs.get('optimizer').state_dict(),
            }

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


