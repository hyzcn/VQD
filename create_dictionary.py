#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:30:29 2018

@author: manojacharya
"""
import numpy as np
from config import dataset
from models.dictionary import Dictionary

#this verifies what I am doing is right
# so create dictionary for train and test into a combined
# the test embeddings will  not be updated during train 
#https://github.com/allenai/allennlp/issues/516

def create_dictionary(entries,dataset):
    dictionary = Dictionary()   
    for ent in entries:
        if dataset == 'vqd':
            qs = ent['question']
        else:
            qs = ent['sentence']['sent']
        dictionary.tokenize(qs, True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(val) for val in  vals[1:]]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    import config  
    import os
    import os.path as osp
    import json
    
    for ds in config.dataset:
        kwargs = {**config.global_config , **config.dataset[ds]}
        data_root = kwargs.get('data_root')
        dataset = kwargs.get('dataset')
        splitBy = kwargs.get('splitBy')
        splits = kwargs.get('splits')   
        data = []
        for split in splits  + ['train']:
            data_json = osp.join('cache/prepro', dataset +"_"+ splitBy , split +'.json')
            with open(data_json,'r') as f:
                d = json.load(f)
                data.extend(d)
        
        
        d = create_dictionary(data,dataset=dataset)
        basedir = os.path.dirname(kwargs['dictionaryfile'].format(dataset))
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        d.dump_to_file(kwargs['dictionaryfile'].format(dataset))
        d = Dictionary.load_from_file(kwargs['dictionaryfile'].format(dataset))
        emb_dim = 300
        glove = 'glove/glove.6B.%dd.txt' % emb_dim
        embedding_basedir = os.path.dirname(kwargs['glove'])
        glove_file = embedding_basedir.format(glove)
        weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
        np.save( os.path.join(embedding_basedir.format(ds),'glove6b_init_%dd.npy' % emb_dim), weights)