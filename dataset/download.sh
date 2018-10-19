#!/usr/bin/env bash

# Panoptic annotations
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip -d dataset/.
mv dataset/annotations/panoptic_train2017.json dataset/panoptic_train2017.json
mv dataset/annotations/panoptic_val2017.json dataset/panoptic_val2017.json
rm panoptic_annotations_trainval2017.zip
rm -rf dataset/annotations/