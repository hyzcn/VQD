import json
from utils import *

def create_category_json(label_f):
    """
    Generate a JSON file with category-id as key and
    category name as value.
    :param label_f: MS-COCO labels filepath
    :return: None
    """
    result = dict()
    dest_file_path = '../dataset/panoptic_categories.json'
    result['coco'] = {}
    result['coco-stuff'] = {}
    with open(label_f, 'r') as fp:
        for line in fp:
            id_name = line.strip().split(": ")
            if int(id_name[0]) <=80:
                result['coco'][int(id_name[0])] = id_name[1]
            else:
                result['coco-stuff'][int(id_name[0])] = id_name[1]

    result['info'] = {'key': 'id', 'value': 'name'}
    with open(dest_file_path, 'w') as fp:
        json.dump(result, fp)
    print("DONE! - Generated " + dest_file_path)


def create_image_json(train_f, val_f):
    """
    Generate a JSON file which contains COCO image id as key
    and value represents information about the image such as
    URL, width, height, and others.
    :param train_f: Panoptic train json file path
    :param val_f: Panoptic validation json file path
    :return: None
    """
    js_train = json.load(open(train_f))
    js_val = json.load(open(val_f))
    dest_file_path = '../dataset/panoptic_images.json'

    result = {}
    img_train = js_train['images']
    img_val = js_val['images']

    img_id_to_stats = {}
    for img_dict in img_train:
        coco_img_id = img_dict['id']
        img_dict['split'] = 'train'
        img_id_to_stats[int(coco_img_id)] = img_dict

    for img_dict in img_val:
        coco_img_id = img_dict['id']
        img_dict['split'] = 'val'
        img_id_to_stats[int(coco_img_id)] = img_dict

    result['info'] = {'key': 'coco_img_id', 'value': 'Image statistics'}
    result['images'] = img_id_to_stats

    with open(dest_file_path, 'w') as fp:
        json.dump(result, fp)
    print("DONE! - Generated " + dest_file_path)


def create_annotations_json(train_f, val_f):
    """
    Generate a JSON file contains image annotations such as
    bounding box, category id, image id, etc.
    :param train_f: Panoptic train json file path
    :param val_f: Panoptic validation json file path
    :return: None
    """
    js_train = json.load(open(train_f))
    js_val = json.load(open(val_f))
    dest_file_path = '../dataset/panoptic_annotations.json'

    result = {}
    ann_train = js_train['annotations']
    ann_val = js_val['annotations']

    img_id_to_ann = {}
    for ann_dict in ann_train:
        coco_img_id = ann_dict['image_id']
        img_id_to_ann[int(coco_img_id)] = ann_dict

    for ann_dict in ann_val:
        coco_img_id = ann_dict['image_id']
        img_id_to_ann[int(coco_img_id)] = ann_dict

    result['info'] = {'key': 'coco_img_id', 'value': 'Annotations'}
    result['annotations'] = img_id_to_ann

    with open(dest_file_path, 'w') as fp:
        json.dump(result, fp)
    print("DONE! - Generated " + dest_file_path)


if __name__ == '__main__':
    train_f = '../dataset/panoptic_train2017.json'
    val_f = '../dataset/panoptic_val2017.json'
    label_f = '../dataset/labels.txt'
    create_category_json(label_f)
    create_image_json(train_f, val_f)
    create_annotations_json(train_f, val_f)
