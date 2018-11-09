#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""

import torch
import torch.nn as nn
from .language import QuestionParser

class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()

        I_CNN = 2048
        Q_GRU_out = 1024
        Q_embedding = 300
        LINsize = 1024
        Boxcoords = 16

        self.Ncls = 2 #either true to false 
        
     
        self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.0, word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')
            

        insize = Boxcoords +  I_CNN * 2  + Q_GRU_out

        Wlayers =        [nn.Linear(insize, LINsize),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.5),
                           nn.Linear(LINsize,1)]
    
        self.fscore = nn.Sequential(*Wlayers) 
     
        fcls_layers =  [ nn.Linear( insize, LINsize),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.5),
                           nn.Linear( LINsize, LINsize),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.5),
                           nn.Linear(LINsize,self.Ncls)]

        self.fcls = nn.Sequential(*fcls_layers) 
        self.device = next(self.fcls.parameters()).device
        #self.device = torch.device("cuda")

    @staticmethod
    def get_spatials(b):
        # b = (B, k, 6)

        b = b.float()

        k, _ = b.size()
        

        b_ij = b.unsqueeze(1).expand(-1,k,-1)  # (k, k, 6)
        b_ji = b.unsqueeze(0).expand(k,-1,-1)

        area_ij = (b_ij[..., 2] - b_ij[..., 0]) * (b_ij[..., 3] - b_ij[..., 1])
        area_ji = (b_ji[..., 2] - b_ji[..., 0]) * (b_ji[..., 3] - b_ji[..., 1])

        righmost_left = torch.max(b_ij[..., 0], b_ji[..., 0])
        downmost_top = torch.max(b_ij[..., 1], b_ji[..., 1])
        leftmost_right = torch.min(b_ij[..., 2], b_ji[..., 2])
        topmost_down = torch.min(b_ij[..., 3], b_ji[..., 3])

        # calucate the separations
        left_right = (leftmost_right - righmost_left)
        up_down = (topmost_down - downmost_top)

        # don't multiply negative separations,
        # might actually give a postive area that doesn't exit!
        left_right = torch.max(0*left_right, left_right)
        up_down = torch.max(0*up_down, up_down)

        overlap = left_right * up_down
        
        eps = 1e-6
        #division by zero may cause nans in gradient

        iou = overlap / (area_ij + eps + area_ji - overlap)
        o_ij = overlap / (area_ij + eps)
        o_ji = overlap / (area_ji + eps)

        iou = iou.unsqueeze(-1)  # (k, k, 1)
        o_ij = o_ij.unsqueeze(-1)  # (k, k, 1)
        o_ji = o_ji.unsqueeze(-1)  # (k, k, 1)

        return b_ij, b_ji, iou, o_ij, o_ji


    def forward(self,box_feats,q_feats,box_coords,index):

        q_rnn  = self.QRNN(q_feats)
        b,d,k = box_feats.size()
        
        logits = []
        scores = []
        for i in range(b):

            #N =  int(index[i]) # number of boxes
            N = 100
            
            box_feats_idx = box_feats[i,:N,:]
            q_rnn_idx  =  q_rnn[i,:].unsqueeze(0)
            box_coords_idx = box_coords[i,:N,:]
            

            qst = q_rnn_idx.unsqueeze(1).expand(N,N,-1)
            
            o_i = box_feats_idx.unsqueeze(1).expand(-1,N,-1)
            o_j = box_feats_idx.unsqueeze(0).expand(N,-1,-1)
            
            # dot product: (B, k, k)
            vtv = torch.mul(o_i.contiguous().view(N*N,-1) , o_j.contiguous().view(N*N,-1))
            vtv = torch.sum(vtv,dim=1)
            dot = vtv.view(N,N,-1)
    
            b_ij, b_ji, iou, o_ij, o_ji = self.get_spatials(box_coords_idx)                        
            features = [ dot, b_ij , b_ji ,iou ,o_ij, o_ji]  # (k, k, 6)
            features = torch.cat(features, dim=-1)  # ( k, k, 16)
            
            b_full = torch.cat([o_i,o_j,qst,features],-1)
            b_full = b_full.view(N*N,-1)
            
            score = self.fscore(b_full)
            score = score.view(N,N,-1)
            logit  = self.fcls(b_full)
            logit = logit.view(N,N,-1)
            
            
            scoremax,idxmax = torch.max(score.cpu(),dim=1)    
            index = torch.tensor(range(0,N)).unsqueeze(1).long()
            index = index.to(self.device)
            idxmax = idxmax.to(self.device)
            ii = torch.cat((index,idxmax),dim=1)                        
            score_sel = score[ii[:,0],ii[:,1],:]
            logit_sel = logit[ii[:,0],ii[:,1],:]
                       
#            scoresel = torch.gather(score,1,idxmax.unsqueeze(2)).squeeze(-1)
#            logitsel = torch.gather(logit,1,idxmax.unsqueeze(2).repeat(1,1,2))
#            logitsel = logitsel.squeeze(1)
            
            logits.append(logit_sel.unsqueeze(0))
            scores.append(score_sel.unsqueeze(0))
        finscores = torch.cat(scores,0)
        finlogits = torch.cat(logits,0)
        return finscores,finlogits