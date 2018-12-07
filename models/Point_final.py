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
        Boxcoords = 6

        self.Ncls = 2 #either true to false 
        
        bidirectional = True
        
#        if bidirectional:
#            insize = Boxcoords +  I_CNN + Q_GRU_out*2 + I_CNN
#        else:
#            insize = Boxcoords +  I_CNN + Q_GRU_out + I_CNN
        
        
        if bidirectional:
            grusize = Q_GRU_out*2 
        else:
            grusize = Q_GRU_out
            
            
        insize = Boxcoords + I_CNN + grusize 
      
        self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.3,
                                         bidirectional=bidirectional,
                                         word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')
               
        
        Nboxes = 45
        
        common_layers =   [nn.Linear(insize, LINsize),
                           nn.ReLU(inplace=True),
                           #nn.Dropout(0.5),
                           nn.Linear(LINsize,LINsize),
                            nn.ReLU(inplace=True),
                           #nn.Dropout(0.5),
                           nn.Linear(LINsize,LINsize)]   
        self.fcommon = nn.Sequential(*common_layers) 
        
        #B x 100 x 1024         
        #self.bn2 = nn.BatchNorm1d(num_features=Nboxes)
        
        fscore_layers =  [ nn.ReLU(inplace=True),
                           nn.Linear( LINsize + I_CNN ,1)]
        
        
        self.fscore = nn.Sequential(*fscore_layers) 

        #fcls_layers =   [ nn.ReLU(inplace=True),
        #                   nn.Linear(2 * I_CNN + 1024,self.Ncls)]
        
        
        fcls_layers =   [ nn.ReLU(inplace=True),
                           nn.Linear( LINsize +  I_CNN, 2)]

        self.fcls = nn.Sequential(*fcls_layers) 
        
        
   
        
        hid_dim = 512
        # gated tanh activation
        self.gt_W_img_att = nn.Linear(2 * I_CNN + Boxcoords, hid_dim)
        self.gt_W_prime_img_att = nn.Linear(2 * I_CNN+ Boxcoords, hid_dim)
        
                # image attention
        self.att_wa = nn.Linear(hid_dim, 1)  
        
        

    def forward(self,box_feats,q_feats,box_coords,index):

        q_rnn  = self.QRNN(q_feats)
        b,d,k = box_feats.size()
        qst  =  q_rnn.unsqueeze(1)
        qst = qst.repeat(1, d, 1) 
        b_full = torch.cat([box_feats,qst,box_coords],-1)           
        common = self.fcommon(b_full)

        
        
        concated = self._gated_tanh(b_full, self.gt_W_img_att, self.gt_W_prime_img_att)
        
        awa = self.att_wa(concated)
        awa = torch.softmax(awa.squeeze(),dim=-1)
        context = torch.bmm(awa.unsqueeze(1), box_feats).squeeze()
        
        context = context.unsqueeze(1).expand(-1,d,-1)
        common_latefusion = torch.cat([common,context],-1)  
        batch_context = common_latefusion
        #batch_context = self.bn2(common_latefusion)

        
        scores = self.fscore(batch_context)
        # dont know why clone is needed here
        #backward was shgowing some error
        logits =  self.fcls(batch_context.clone()) 
        return  scores,logits

    def _gated_tanh(self, x, W, W_prime):
        """
        Implement the gated hyperbolic tangent non-linear activation
            x: input tensor of dimension m
        """
        y_tilde = torch.tanh(W(x))
        g = torch.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y
