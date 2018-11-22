#!/usr/bin/env bash

# Run the `dataset/download.sh` file to download all
# the annotations required to construct a VQD dataset.
#sh download.sh

# `process_panoptic.py`to segregate the panoptic image, annotation,
# and category which helps in processing the meta data faster.
python process_panoptic.py

# `process_visual_genome.py` to get the image annotations
# which helps in processing the meta data faster.
python process_visual_genome.py

# `object_detection_questions.py` generates and save the simple
# object detection (Question, bounding boxes) pair to VQD dataset
# with the help of coco panoptic annotations.
python object_detection_questions.py

# `color_reasoning_questions.py` generates and save the Color
# reasoning (Question, bounding boxes) pair to VQD dataset with
# the help of Visual Genome annotations.
python color_reasoning_questions.py

# `positional_reasoning_questions.py` generates and save the Positional
# reasoning (Question, bounding boxes) pair to VQD dataset with
# the help of Visual Genome annotations.
python positional_reasoning_questions.py

# `absurd_questions.py` generates a question about an object which is
# not present in the image. It saves (Question, Empty bounding boxes)
# pair to VQD dataset
python absurd_questions.py

# It converts the MS-COCO 2017 train/val split images to MS-COCO 2014
# train/val split. Don't run the below line if you need MS-COCO 2017
# train/val split
python dataset_split_ratio.py

# `create_vqd_json.py` creates a final train and val VQD json
# annotation file.
python create_vqd_json.py
