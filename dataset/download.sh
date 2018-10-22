#!/usr/bin/env bash

# Panoptic annotations
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip -d dataset/.
mv dataset/annotations/panoptic_train2017.json dataset/panoptic_train2017.json
mv dataset/annotations/panoptic_val2017.json dataset/panoptic_val2017.json
rm panoptic_annotations_trainval2017.zip
rm -rf dataset/annotations/

# Visual Genome annotations
wget https://visualgenome.org/static/data/dataset/attributes.json.zip
unzip attributes.json.zip -d dataset/.
rm attributes.json.zip

wget https://visualgenome.org/static/data/dataset/image_data.json.zip
unzip image_data.json.zip -d dataset/.
rm image_data.json.zip