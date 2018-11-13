#!/usr/bin/env bash

mkdir dataset/

# Panoptic annotations
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip -d dataset/
mv dataset/annotations/panoptic_*.json dataset/


# Visual Genome annotations
wget https://visualgenome.org/static/data/dataset/attributes.json.zip
unzip attributes.json.zip -d dataset/


wget https://visualgenome.org/static/data/dataset/image_data.json.zip
unzip image_data.json.zip -d dataset/


wget https://visualgenome.org/static/data/dataset/relationships_v1_2.json.zip
unzip relationships_v1_2.json.zip -d dataset/

