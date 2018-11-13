import json
from utils import *


def create_category_json(train_f):
    """
    Generate a JSON file with `things` and `stuff` label
    and their list of supercategories to generate a tree
    structure
    Panoptic train and val has same categories, so used just
    train categories annotations
    :param train_f: Panoptic train json file path
    :return: None
    """
    things_label = dict()  # To store COCO things labels as {id:name}
    stuff_label = dict()  # To store COCO stuff labels as {id:name}
    t_tree = []  # list of CHILD:PARENT name for things
    s_tree = []  # list of CHILD:PARENT name for stuff
    dest_file_path = 'dataset/panoptic_categories.json'

    categories = json.load(open(train_f))['categories']
    for categ in categories:
        if categ['isthing'] == 1:
            # Store the COCO things
            things_label[categ['id']] = categ['name']
            t_tree.append(categ['name'] + ":" + categ['supercategory'])
        else:
            # Store the COCO stuff
            stuff_label[categ['id']] = categ['name']
            s_tree.append(categ['name'] + ":" + categ['supercategory'])

    things_outdoor_scatg = ['sports', 'accessory', 'animal', 'outdoor', 'vehicle', 'person']
    things_indoor_scatg = ['indoor', 'appliance', 'electronic', 'furniture', 'food', 'kitchen']
    stuff_outdoor_scatg = ['water', 'ground', 'solid', 'sky', 'plant', 'structural', 'building']
    stuff_indoor_scatg = ['food', 'textile', 'furniture', 'window', 'floor', 'ceiling', 'wall', 'rawmaterial']

    things_outdoor = [child_name + ':OUTDOOR' for child_name in things_outdoor_scatg]
    things_indoor = [child_name + ':INDOOR' for child_name in things_indoor_scatg]
    stuff_outdoor = [child_name + ':OUTDOOR' for child_name in stuff_outdoor_scatg]
    stuff_indoor = [child_name + ':INDOOR' for child_name in stuff_indoor_scatg]

    # Create a CHILD:PARENT mapping which will be useful in constructing
    # a level-wise tree
    things_tree = ['CHILD:PARENT'] + ['things:None'] + \
                  ['OUTDOOR:things'] + ['INDOOR:things'] + \
                  things_outdoor + things_indoor + t_tree
    stuff_tree = ['CHILD:PARENT'] + ['stuff:None'] + \
                 ['OUTDOOR:stuff'] + ['INDOOR:stuff'] + \
                 stuff_outdoor + stuff_indoor + s_tree

    output_dict = dict()
    output_dict['things'] = {
        'label': things_label,
        'tree': things_tree
    }
    output_dict['stuff'] = {
        'label': stuff_label,
        'tree': stuff_tree
    }

    with open(dest_file_path, 'w') as fp:
        json.dump(output_dict, fp)
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
    dest_file_path = 'dataset/panoptic_images.json'

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
    dest_file_path = 'dataset/panoptic_annotations.json'

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
    train_f = 'dataset/panoptic_train2017.json'
    val_f = 'dataset/panoptic_val2017.json'
    create_category_json(train_f)
    create_image_json(train_f, val_f)
    create_annotations_json(train_f, val_f)
