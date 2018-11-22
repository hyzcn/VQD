#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:42:10 2018

@author: manoj
"""
from utils import parsejson,dumpjson

def getqid2box(data):
    qid2box = {}
    for ent in data['Annotations']:    
        boxes = ent['question_id_bbox']
        for qid in boxes:
            qid2box[qid] = boxes[qid]
                
    return qid2box    

def getdata(annotations,start):
    "starting piont of qid"
    qidcount = start
    data = []
    for annt in annotations:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            ent = {}        
            ques_dict = qid2questions[ques_id]
            ques_type = ques_dict['question_type']
            ques = ques_dict['question']
            ent['gtbox'] = bbox
            ent['question'] = ques
            ent['width'] = annt['coco_width']
            ent['vqd_question_id'] = ques_id
            ent['question_type'] = ques_type
            ent['height'] = annt['coco_height']
            ent['image_id'] = int(annt['coco_image_id'])
            ent['question_id'] = int(qidcount)
            data.append(ent)           
            qidcount +=1
    return data

#%%
    
val = parsejson('/home/manoj/vqd/VQD_dataset_creation/VQD/vqd_val.json')
train = parsejson('/home/manoj/vqd/VQD_dataset_creation/VQD/vqd_train.json')
print (val.keys())

questions = train['Question_id']
qid2box = getqid2box(questions)
train_annotations = train['Annotations']
val_annotations = val['Annotations']
qid2questions = { ent['question_id']:ent for ent in questions}
#%%

train_data = getdata(train_annotations,start=0)
val_data = getdata(val_annotations,start=len(train_data))

dumpjson(train_data,'/home/manoj/vqd/cache/prepro/vqd_/train.json')
dumpjson(val_data,'/home/manoj/vqd/cache/prepro/vqd_/val.json')

#%%
from collections import defaultdict
def getstats(data):
    data_type = defaultdict(int)
    data_type['total'] = len(data)
    for qent in data:
        qtype = qent['question_type']
        data_type[qtype]+=1
    print(data_type)
    
getstats(train_data)
getstats(val_data)