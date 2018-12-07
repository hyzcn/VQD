#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""
import torch.nn as nn


class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()

        I_CNN = 2048
        LINsize = 1024
        self.Ncls = 2 #either true to false 
        insize = I_CNN
        
        common_layers =   [nn.Linear(insize, LINsize),
                           nn.ReLU(inplace=True),
                           nn.Linear(LINsize,LINsize)]
    
        self.fcommon = nn.Sequential(*common_layers) 
        
        
        fscore_layers =  [ nn.ReLU(inplace=True),
                           nn.Linear(LINsize,1)]
        
        
        self.fscore = nn.Sequential(*fscore_layers) 

        fcls_layers =  [   nn.ReLU(inplace=True),
                           nn.Linear(LINsize,self.Ncls)]

        self.fcls = nn.Sequential(*fcls_layers) 

    def forward(self,box_feats,q_feats,box_coords,index):

        b,d,k = box_feats.size()          
        common = self.fcommon(box_feats)
        scores = self.fscore(common)
        # dont know why clone is needed here
        #backward was shgowing some error
        logits =  self.fcls(common.clone()) 
        return  scores,logits