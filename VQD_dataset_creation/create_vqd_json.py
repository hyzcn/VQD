import json
import os

def question_tokenize(questions_bbox):
    """
    Map unique question to index and index to question
    :param questions_bbox: List of question bounding box annotations
    :return: A tuple containing tokenize question
    """
    question2idx = dict()
    idx2question = []

    index = 0
    for ques_bbox_dict in questions_bbox:
        for ques, bbox_ques_type in ques_bbox_dict.items():
            ques_type = bbox_ques_type[1]

            if ques not in question2idx:
                # Unique question gets a unique index
                question2idx[ques] = str(index)

                # Each list index saves a dictionary of question,
                # question_id, and question_type
                idx2question.append({
                    'question': ques,
                    'question_id': str(index),
                    'question_type': ques_type
                })
                index += 1
    return question2idx, idx2question


def image_tokenize(coco_id):
    """
    Map MS-COCO image id to index and vice-versa
    :param coco_id: List of sorted MS-COCO image id
    :return: A tuple containing tokenize image id
    """
    coco_id2idx = dict()

    for i, image_id in enumerate(coco_id):
        coco_id2idx[str(image_id)] = i
        coco_id[i] = str(image_id)
    return coco_id2idx, coco_id


def main():
    """
    It generates a final VQD json file
    :return: None
    """
    file_name = 'dataset/vqd_annotations.json'
    vqd = json.load(open(file_name))
    annotations = vqd['annotations']

    coco_ids = []
    questions_bbox = []

    # Get the list of all MS-COCO image id and
    # questions to bounding box
    for image_id, stats in annotations.items():
        coco_ids.append(image_id)
        questions_bbox.append(stats['question_id_bbox'])

    # Sort the MS-COCO image id in increasing order
    coco_ids.sort()

    # Get the tokenize question and MS-COCO image id
    question2idx, idx2question = question_tokenize(questions_bbox)
    coco_id2idx, idx2coco_id = image_tokenize(coco_ids)

    # Transform the question-to-bounding box to question-id-to-bounding box
    for image_id, stats in annotations.items():
        ques_bbox = stats['question_id_bbox']
        new_ques_bbox = dict()
        for ques, bbox_ques_type in ques_bbox.items():
            new_ques_bbox[question2idx[ques]] = bbox_ques_type[0]
        stats['question_id_bbox'] = new_ques_bbox

    # Convert the annotation dictionary to a list
    new_annotation_list = []
    for image_id in idx2coco_id:
        stats = annotations[image_id]
        new_annotation_list.append(stats)

    # Split the annotations into train and val
    train_annotations = []
    val_annotations = []
    for stats in new_annotation_list:
        split = stats['split']
        if split == 'train':
            train_annotations.append(stats)
        else:
            val_annotations.append(stats)

    # Create separate train and val dictionary
    train_vqd_json = dict()
    val_vqd_json = dict()

    train_vqd_json['Data_subtype'] = 'train'
    train_vqd_json['Data_type'] = 'mscoco and visual_genome'
    train_vqd_json['Annotations'] = train_annotations
    train_vqd_json['Question_id'] = idx2question

    val_vqd_json['Data_subtype'] = 'val'
    val_vqd_json['Data_type'] = 'mscoco and visual_genome'
    val_vqd_json['Annotations'] = val_annotations
    val_vqd_json['Question_id'] = idx2question


    # Write to a file
    if not os.path.exists("VQD"):
        os.mkdir("VQD")    
    train_vqd_file_path = 'VQD/vqd_train.json'
    val_vqd_file_path = 'VQD/vqd_val.json'
    with open(train_vqd_file_path, 'w') as fp:
        json.dump(train_vqd_json, fp)
    with open(val_vqd_file_path, 'w') as fp:
        json.dump(val_vqd_json, fp)


if __name__ == '__main__':
    main()
