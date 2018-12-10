#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""



#concat(v,q) -> bnorm -> 1024 -> 1024 -> 1024 -> 1024

#That produces 1024 dimensional "multi-modal embedding" for each object m

#concat(m, q) -> bnorm -> bigru -> classifier




import torch
import torch.nn as nn
from .language import QuestionParser

class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()
        self.__name__ = 'final'

        I_CNN = 2048
        Q_GRU_out = 1024
        Q_embedding = 300
        LINsize = 1024

        
        bidirectional = True        
        if bidirectional:
            grusize = Q_GRU_out*2 
        else:
            grusize = Q_GRU_out
            
            
        insize = I_CNN + grusize 
      
        self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.0,
                                         bidirectional=bidirectional,
                                         word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')
               

        common_layers =   [nn.Linear(insize, LINsize),
                           nn.ReLU(inplace=True),
                           nn.Linear(LINsize,LINsize),]   
        self.fcommon = nn.Sequential(*common_layers) 
        
        
        fscore_layers =  [ nn.ReLU(inplace=True),
                           nn.Linear( LINsize ,1)]
        
        
        self.fscore = nn.Sequential(*fscore_layers) 

       
        
        fcls_layers =   [ nn.ReLU(inplace=True),
                           nn.Linear( LINsize, 2)]

        self.fcls = nn.Sequential(*fcls_layers) 
        
        

#    def forward(self,box_feats,q_feats,box_coords,index):
#
#        q_rnn  = self.QRNN(q_feats)
#        b,d,k = box_feats.size()
#        qst  =  q_rnn.unsqueeze(1)
#        qst = qst.repeat(1, d, 1) 
#
#        b_full = torch.cat([box_feats,qst],-1)            
#        common = self.fcommon(b_full)
#
#        scores = self.fscore(common)
#        # dont know why clone is needed here
#        #backward was shgowing some error
#        logits =  self.fcls(common.clone()) 
#        return  scores,logits


    def forward(self,box_feats,q_feats,box_coords,index):

        q_rnn  = self.QRNN(q_feats)
        b,d,k = box_feats.size()
        qst  =  q_rnn.unsqueeze(1)
        qst = qst.repeat(1, d, 1) 

        b_full = torch.cat([box_feats,qst],-1)            
        common = self.fcommon(b_full)

        scores = self.fscore(common)
        scores = torch.sigmoid(scores)
        # dont know why clone is needed here
        #backward was shgowing some error
        logits =  self.fcls(common.clone()) 
        return  scores,logits
