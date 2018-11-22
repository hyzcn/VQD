#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:36:51 2018

@author: manoj
"""

import torch
import numpy as np
import torch.nn as nn
from .dictionary import Dictionary

def tokenize_ques(dictionary,question,max_length=14):
    """Tokenizes the questions.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """

    tokens = dictionary.tokenize(question, add_word = None)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    assert len(tokens) ==  max_length , "Tokens NOT EQUAL TO MAX length."
    return np.array(tokens)

class QuestionParser(nn.Module):
    def __init__(self,dictionaryfile = None, glove_file = None,bidirectional=False,
                 dropout=0.3, word_dim=300, ques_dim=1024 , rnn_type= 'GRU'):
        super().__init__()

        self.dropout = dropout
        self.word_dim = word_dim
        self.ques_dim = ques_dim
        self.bidirectional = bidirectional

        dictionary = Dictionary.load_from_file(dictionaryfile)
        self.VOCAB_SIZE = dictionary.ntoken
        self.glove_file = glove_file

        self.embd = nn.Embedding(self.VOCAB_SIZE + 1, self.word_dim,
                                 padding_idx=self.VOCAB_SIZE)
        
        if rnn_type == 'GRU':
            RNN  = nn.GRU
        else:
            RNN = nn.LSTM
        
        self.rnn = RNN(input_size = self.word_dim, hidden_size=self.ques_dim,
                       bidirectional = self.bidirectional)
        self.drop = nn.Dropout(self.dropout)
        self.glove_init()

        if self.bidirectional:
            self.forward = self.forward_bidir
        else:
            self.forward = self.forward_unidir

    def glove_init(self):
        print("initialising with glove embeddings")
        glove_embds = torch.from_numpy(np.load(self.glove_file))
        assert glove_embds.size() == (self.VOCAB_SIZE, self.word_dim)
        self.embd.weight.data[:self.VOCAB_SIZE] = glove_embds
        print("done..")

    def forward_unidir(self, questions):
        # (B, MAXLEN)
        # print("question size ", questions.size())
        questions = questions.t()  # (MAXLEN, B)
        questions = self.embd(questions)  # (MAXLEN, B, word_size)
        _, (q_emb) = self.rnn(questions)
        q_emb = q_emb[-1]  # (B, ques_size)
        q_emb = self.drop(q_emb)
        return q_emb
    
    def forward_bidir(self, questions):
        
        #Refer this 
        #https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        # (B, MAXLEN)
        # print("question size ", questions.size())
        questions = questions.t()  # (MAXLEN, B)
        questions = self.embd(questions)  # (MAXLEN, B, word_size)
        output, h_n = self.rnn(questions)
        q_emb_forward = output[-1, :, :self.ques_dim]
        q_emb_reverse = output[0, :, self.ques_dim:]
        q_emb = torch.cat((q_emb_forward,q_emb_reverse),dim=-1)
        q_emb = self.drop(q_emb)
        return q_emb

#%%
    
if __name__ == '__main__':  

    import sys
    sys.path.append("../")
    import config   
    import os.path as osp
    dictfile = config.global_config.get('dictionaryfile').format('refcoco')
    d = Dictionary.load_from_file(osp.join("..",dictfile))
    
    tokens = tokenize_ques(d,"dogs left to the tree")
    question = torch.from_numpy(tokens)
    question = question.unsqueeze(0)    
    print (question)
