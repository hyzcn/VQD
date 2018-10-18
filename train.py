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
from CLR import CyclicLR
from utils import adjust_learning_rate
#%%


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def main(**kwargs):
    
    device = kwargs.get('device') 
    net = kwargs.get('model')                     
    optimizer = kwargs.get('optimizer')
    epoch = kwargs.get('epoch')
    istrain = kwargs.get('istrain')
  
    start_time = time.time()
    true = []
    pred_reg = []
    pred_cls = []
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

    reglossfn = nn.SmoothL1Loss() # also known as huber loss
    #reglossfn = nn.MSELoss()
    #reglossfn = nn.L1Loss()
    #clslossfn = nn.CrossEntropyLoss()
    clslossfn = instance_bce_with_logits
    
    with torch.set_grad_enabled(istrain):
        for i, data in enumerate(loader):
            qid,wholefeat,pooled,boxes,labels,targets,ques,box_coords,index = data            
            idxs.extend(qid.tolist())        
            labels = labels.long()            
            index  = index.long()
            B = qid.size(0)
            #converts 14_14 to 7_7
            #change pool size
            
            if torch.sum(pooled):
                pooled = F.avg_pool2d(pooled.permute(0,3,1,2),8,2)
                Npool = pooled.size(-1)
                pooled = pooled.view(B,2048,Npool**2)
                pooled = pooled.permute(0,2,1)
                pooled = F.normalize(pooled,p=2,dim=-1)
                #print (pooled.shape)
    
                pooled = pooled.to(device)
                wholefeat = F.normalize(wholefeat,p=2,dim=-1)
            else:
                pooled = wholefeat = None
   
            true.extend(labels.tolist())
    
            #normalize the box feats
            boxes = F.normalize(boxes,p=2,dim=-1)
            box_feats = boxes.to(device)
            box_coords = box_coords.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            q_feats = ques.to(device)
     
            optimizer.zero_grad()
            
            net_kwargs = { 'wholefeat':wholefeat,
                           'pooled' :pooled,
                           'box_feats':box_feats,
                           'q_feats':q_feats,
                           'box_coords':box_coords,
                           'index':index}

            out = net(**net_kwargs)
                          
            if out.ndimension() == 1:  # if regression
                loss = reglossfn(out,labels.float())
                #round the output
                regpred = torch.round(out.data.cpu()).numpy().ravel()
                pred_reg.extend(regpred)
    
    
            else: # if classification
                #loss = clslossfn(out, labels.long())
                loss = clslossfn(out, targets)
                _,clspred = torch.max(out,-1)
                pred_reg.extend(clspred.data.cpu().numpy().ravel())
        
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
        ent['pred_reg'] = pred_reg
        ent['pred_cls'] = pred_reg
        ent['loss'] = loss_meter.avg
        ent['qids'] = idxs
        return ent



def run(**kwargs):

    savefolder = kwargs.get('savefolder')
    logger = kwargs.get('logger')
    epochs = kwargs.get('epochs')
    isVQAeval = kwargs.get('isVQAeval')
    N_classes = kwargs.get('N_classes')
    test_loader = kwargs.get('test_loader')
    start_epoch = kwargs.get('start_epoch')
    eval_baselines = kwargs.get('nobaselines') == False
    #DETECT, MUTAN , Zhang , UPdown baselines

    if start_epoch == 0 and eval_baselines: # if not resuming
        pass
        #eval_extra.main(**kwargs)
        
    testset = test_loader.dataset.data
    early_stop = EarlyStopping(monitor='loss',patience=8)
    
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=1000., mode='triangular2')
    
    Modelsavefreq = 1

    for epoch in range(start_epoch,epochs):

        kwargs['epoch'] = epoch
        start_time = time.time()
        train = main(istrain=True,**kwargs)
        test =  main(istrain=False,**kwargs)
        total_time  = time.time() - start_time

        logger.write('Epoch {} Time {:2.2f} s ------'.format(epoch,total_time))
        logger.write('\tTrain Loss: {:.4f}'.format(train['loss']))
        logger.append('train_losses',train['loss'])
        logger.write('\tTest Loss: {:.4f}'.format(test['loss']))
        logger.append('test_losses',test['loss'])
                
        if kwargs.get('dsname') == 'VQA2':
            predictions = dict(zip(test['qids'] , test['pred_reg']))
        else:
            pred_reg = np.array(test['pred_reg'],dtype=np.uint64)
            #clamp all output
            pred_reg_clip = pred_reg.clip(min=0,max=N_classes-1).tolist()
            predictions = dict(zip(test['qids'] , pred_reg_clip))
                     
        if isVQAeval:
            acc,rmse = eval_extra.evalvqa(testset,predictions,isVQAeval)
            logger.write("\tRMSE:{:.2f} Accuracy {:.2f}%".format(rmse,acc))
          
        else:            
            simp_comp = eval_extra.eval_simp_comp(testset,predictions)
            for d in ['simple','complex']:
                acc,rmse = simp_comp[d]
                logger.write("\t{} RMSE:{:.2f} Accuracy {:.2f}%".format(d,rmse,acc))
            
        if kwargs.get('savejson'):
            js = []
            for qid in predictions:
                ent = {}
                ent["question_id"] = int(qid)
                ent["answer"] = str(predictions[qid])
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
                'pred_reg':test['pred_reg'],
                'pred_cls':test['pred_cls'],
                'qids': test['qids'],
                'optimizer' : kwargs.get('optimizer').state_dict(),
            }

            save_checkpoint(savefolder,tbs,is_best)

        logger.dump_info()
        
#        clr.clr_iterations = (epoch+1)* 1000
#        adjust_learning_rate(kwargs.get('optimizer'), clr.nextlr())
#        lr =  kwargs.get('optimizer').param_groups[0]['lr']
#        logger.write("New Learning rate: {} ".format(lr))
        
        early_stop.on_epoch_end(epoch,logs=test)
        if early_stop.stop_training:
            lr =  kwargs.get('optimizer').param_groups[0]['lr']
            adjust_learning_rate(kwargs.get('optimizer'), lr* 0.8)
            logger.write("New Learning rate: {} ".format(lr))
            early_stop.reset()
            #break
            
    logger.write('Finished Training')


