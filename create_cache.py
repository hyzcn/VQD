#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:30:29 2018

@author: manojacharya
"""
import config   
import os
import os.path as osp
from refer import REFER
import copy
import json
    
def create_cache(**kwargs):
   
    data_root = kwargs.get('data_root')
    dataset = kwargs.get('dataset')
    splitBy = kwargs.get('splitBy')
    splits = kwargs.get('splits')
    refer = REFER(data_root, dataset, splitBy)
           
    # print stats about the given dataset
    print ('dataset [%s_%s] contains: ' % (dataset, splitBy))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print ('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))
    
    checkpoint_dir = osp.join('cache','prepro', ds +"_"+ splitBy)
    if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

    for split in splits + ['train']:
        ref_ids = refer.getRefIds(split=split)
        print ('%s refs are in split [%s].' % (len(ref_ids), split))
        #have to sample various sentences and their tokens from here.    
        data = []
        for ref_id in ref_ids:
            ref = refer.Refs[ref_id]
            image_id = ref['image_id']
            ref['image_info'] = refer.Imgs[image_id]
            sentences = ref.pop('sentences')
            ref.pop('sent_ids')
            coco_boxes_info = refer.imgToAnns[image_id]
            coco_boxes = [ box_ann['bbox'] for box_ann in coco_boxes_info]
            gtbox = refer.refToAnn[ref_id]['bbox']
            for sentence in sentences:
                entnew = copy.deepcopy(ref)
                entnew['boxes'] = coco_boxes
                entnew['sentence'] = sentence
                entnew['gtbox'] = gtbox
                data.append(entnew)    
            
        data_json = osp.join('cache/prepro', ds +"_"+ splitBy , split +'.json')
        with open(data_json,'w') as f:
            json.dump(data,f)


if __name__ == '__main__':
    ds = 'refcoco+'
    for ds in config.dataset:
        kwargs = {**config.global_config , **config.dataset[ds]}
        create_cache(**kwargs)