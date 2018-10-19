import json
import os.path


def load_plural():
    """
    It load singular and plural words
    :return: Dictionary representing key as singular word and
    value as plural word
    """
    filename = '../dataset/plural.txt'
    result = {}
    with open(filename, 'r') as fp:
        for line in fp:
            names = line.strip().split(": ")
            result[names[0]] = names[1]
    return result


def get_catg_name_to_bbox(stats, categories):
    """
    Generate a maping of category name to bounding boxes for
    panoptic dataset
    :param stats: Dictionary containing information about bounding box,
                  segmentation info, category id
    :param categories: A dictionary represents category id to name
    :return: A dictionary of category name/label to bounding boxes
    """
    catg_to_bbox = {}
    coco_labels = categories['coco']
    list_of_dict = stats['segments_info']
    for dictnry in list_of_dict:
        cat_id = str(dictnry['category_id'])
        if cat_id in coco_labels:
            if cat_id in catg_to_bbox:
                catg_to_bbox[cat_id].append(dictnry['bbox'])
            else:
                catg_to_bbox[cat_id] = [dictnry['bbox']]

    for cat_id in catg_to_bbox.keys():
        cat_name = coco_labels[str(cat_id)]
        catg_to_bbox[cat_name] = catg_to_bbox.pop(cat_id)

    return catg_to_bbox


def write_to_file(coco_id_to_questions_dict):
    """
    Write the VQD data to a JSON file.
    :param coco_id_to_questions_dict: A dictionary containing mapping of coco
            image id with list of questions and their respective bounding boxes
    :return: None
    """
    output_file = '../dataset/vqd_annotations.json'
    exist = os.path.isfile(output_file)
    if exist:
        print ("TODO")
        pass
    else:
        images = json.load(open('../dataset/panoptic_images.json'))['images']
        output = dict()
        output['annotations'] = dict()
        for coco_img_id, questions_dict in coco_id_to_questions_dict.items():
            image_stat = images[coco_img_id]
            annt_stats = {
                'filename': image_stat['file_name'],
                'coco_url': image_stat['coco_url'],
                'height': image_stat['height'],
                'width': image_stat['width'],
                'split': image_stat['split'],
                'qa': questions_dict
            }

            output['annotations'][coco_img_id] = annt_stats

        with open(output_file, 'w') as fp:
            json.dump(output, fp)
        print("DONE - Generated: " + output_file)
