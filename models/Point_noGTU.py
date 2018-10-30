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
        Boxcoords = 6

        self.Ncls = 2 #either true to false 
        
     
        self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.0, word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')
               

        insize = Boxcoords +  I_CNN + Q_GRU_out

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

    def forward(self,box_feats,q_feats,box_coords):

        q_rnn  = self.QRNN(q_feats)
        b,d,k = box_feats.size()
        qst  =  q_rnn.unsqueeze(1)
        qst = qst.repeat(1, d, 1)        
        b_full = torch.cat([box_feats,qst,box_coords],-1)            
        scores = self.fscore(b_full)
        logits =  self.fcls(b_full) 
        return  scores,logits


