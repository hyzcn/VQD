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
        LINsize = 1024
        self.Ncls = 2 #either true to false 
        insize = I_CNN
        
        common_layers =   [nn.Linear(insize, LINsize),
                           nn.ReLU(inplace=True),
                           nn.Dropout(0.5),
                           nn.Linear(LINsize,LINsize)]
    
        self.fcommon = nn.Sequential(*common_layers) 
        
        #B x 100 x 1024         
        self.bn1 = nn.BatchNorm1d(num_features=100)
        
        fscore_layers =  [ nn.ReLU(inplace=True),
                           nn.Linear(LINsize,1)]
        
        
        self.fscore = nn.Sequential(*fscore_layers) 

        fcls_layers =  [   nn.ReLU(inplace=True),
                           nn.Linear(LINsize,self.Ncls)]

        self.fcls = nn.Sequential(*fcls_layers) 

    def forward(self,box_feats,q_feats,box_coords,index):

        b,d,k = box_feats.size()      
        b_full = box_feats      
        common = self.fcommon(b_full)
        common_batch = self.bn1(common)
        scores = self.fscore(common_batch)
        # dont know why clone is needed here
        #backward was shgowing some error
        logits =  self.fcls(common_batch.clone()) 
        return  scores,logits


