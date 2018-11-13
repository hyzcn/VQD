import json
import os.path
import urllib
import numpy as np
import cv2
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_coco_labels():
    """
    It extracts all the coco labels(things and stuff)
    :return: A set of coco labels
    """
    panop_catg_file_path = 'dataset/panoptic_categories.json'
    categories = json.load(open(panop_catg_file_path))
    coco_thing_label = list(categories['things']['label'].values())
    coco_stuff_label = list(categories['stuff']['label'].values())
    coco_labels = set(coco_thing_label + coco_stuff_label)
    return coco_labels


def transform_vis_bbox_to_coco_bbox(coco_id_ques_dict):
    """
    The image dimension in visual genome and MS-COCO dataset is different
    for the same image. This method converts the coordinates of visual
    genome bounding boxes into a MS-COCO bounding boxes with the help of
    visual genome and MS-COCO image height and width.
    :param coco_id_ques_dict: A dictionary contain coco_id as key and value as
    a question to bounding boxes pair, visual genome image height and width, and
    visual genome url.
    :return: Dictionary with transformed bounding boxes
    """
    new_coco_id_ques_dict = dict()
    panoptic_coco_image_dict = json.load(open('dataset/panoptic_images.json'))['images']
    for coco_id, stats in coco_id_ques_dict.items():
        image_stats = panoptic_coco_image_dict[str(coco_id)]

        coco_h = image_stats['height']
        coco_w = image_stats['width']
        vis_h = stats['vis_height']
        vis_w = stats['vis_width']
        new_w_ratio = coco_w / vis_w
        new_h_ratio = coco_h / vis_h

        qa_dict = stats['question_bbox']
        for ques, bboxes in qa_dict.items():
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                new_x = int(x * new_w_ratio)
                new_y = int(y * new_h_ratio)
                new_w = int(w * new_w_ratio)
                new_h = int(h * new_h_ratio)

                new_bbox = [new_x, new_y, new_w, new_h]
                bboxes[i] = new_bbox
        new_coco_id_ques_dict[coco_id] = {'question_bbox': stats['question_bbox']}
    return new_coco_id_ques_dict


def bb_iou(box_a, box_b, threshold=0.6):
    """
    Check the Intersection Over Union(IOU) of two bounding boxes
    :param box_a: Bounding box coordinates(x, y, w, h)
    :param box_b: Bounding box coordinates(x, y, w, h)
    :param threshold: Threshold value
    :return: True if iou < threshold else false
    """

    # Convert the bounding box coordinates from (x, y, w, h) to
    # (x, y, x1, y1)
    box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
    box_b = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]

    # Get the (x,y) coordinates of an intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Calculate the inter area of two bounding box
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # calculate individual bounding box area
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    if iou > threshold:
        return False
    return True


def draw_bbox(url, qa_dict, coco_id, dataset_type='vqd', idx=0):
    """
    It draws the bounding boxes into an image with questions as title
    of an image
    :param url: An image url
    :param qa_dict: A question to bounding boxes dictionary
    :param coco_id: MS-COCO image id
    :param dataset_type: `vqd` or `panoptic`
    :param idx: index to save images locally
    :return: None
    """

    # Download the image and load it into numpy array
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    image = image[..., ::-1]  # Convert BGR to RGB color space
    ax.imshow(image)
    ques_str = ""
    edge_color = ['blue', 'green', 'red', 'cyan', 'orange',
                  'yellow', 'purple', 'brown', 'gray', 'olive']

    i = 0
    # Same color bounding boxes for each question type
    for ques, bboxes_ques_type in qa_dict.items():
        ec = edge_color[i % len(edge_color)]
        ques_str = ques_str + ques + "(" + str(ec) + ")" + "\n"
        bboxes = bboxes_ques_type[0]

        for bb_coord in bboxes:
            if len(bb_coord) != 0:
                x, y, w, h = bb_coord
                # Create a Rectangle patch
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=ec,
                                         facecolor='none', label="aa")

                # Add the patch to the Axes
                ax.add_patch(rect)
                if dataset_type == 'panoptic':
                    plt.text(x, y, "{}".format(ques), color=ec, fontsize=10)

                # Hide grid lines
                ax.grid(False)

                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
        i += 1
    if dataset_type == 'vqd':
        plt.xlabel(ques_str)

    plt.xticks([])
    plt.yticks([])
    plt.title("coco_id: " + str(coco_id) + " URL: " + str(url))
    plt.show()
    # fig.savefig("Images/original_" + str(idx) + ".png")
    plt.close("all")


def get_catg_name_to_bbox(stats, coco_labels):
    """
    Generate a mapping of category name to bounding boxes for
    panoptic dataset
    :param stats: Dictionary containing information about bounding box,
                  segmentation info, category id
    :param coco_labels: A dictionary of `things` coco labels
    :return: A dictionary of category name/label to bounding boxes
    """
    catg_to_bbox = {}
    catg_name_to_bbox = dict()
    list_of_dict = stats['segments_info']
    for dictnry in list_of_dict:
        cat_id = str(dictnry['category_id'])
        # select categories that is present in coco labels
        if cat_id in coco_labels:
            if cat_id in catg_to_bbox:
                catg_to_bbox[cat_id].append(dictnry['bbox'])
            else:
                catg_to_bbox[cat_id] = [dictnry['bbox']]

    for cat_id in catg_to_bbox.keys():
        cat_name = coco_labels[str(cat_id)]
        catg_name_to_bbox[cat_name] = catg_to_bbox[cat_id]

    return catg_name_to_bbox


def get_unique_categs(stats, coco_labels):
    """
    Get the MS-COCO label names from label or category id
    :param stats: Panoptic Annotation of a single image
    :param coco_labels: label id to name dict
    :return: category names
    """
    catg_names = set()
    list_of_dict = stats['segments_info']
    for dictnry in list_of_dict:
        cat_id = str(dictnry['category_id'])
        # select categories that is present in coco labels
        if cat_id in coco_labels:
            catg_names.add(coco_labels[str(cat_id)])
    return catg_names


def show_n_images(n):
    """
    Display `n` number of images with their respective questions and
    bounding boxes.
    :param n: Number of images to display
    :return: None
    """
    output_file = 'dataset/vqd_annotations.json'
    output = json.load(open(output_file))
    annotations = output['annotations']
    annt_list = list(annotations.keys())
    # Randomly shuffle the list to get the n random images from dataset
    random.shuffle(annt_list)
    if n < len(annt_list):
        annt_list = annt_list[:n]
    else:
        print("Value of n is greater than the list of images in dataset")
        sys.exit(1)

    # Display the first n from the list
    for image_id in annt_list:
        stats = annotations[image_id]
        qa = stats['qa']
        url = stats['coco_url']
        draw_bbox(url, qa, image_id, 'vqd', image_id)


def create_fresh_file(output_file):
    """
    Create a fresh VQD annotation json file without any data
    :param output_file: VQD annotation file path
    :return: None
    """
    images = json.load(open('dataset/panoptic_images.json'))['images']
    output = dict()
    output['annotations'] = dict()

    for img in images:
        annt_stats = {
            'coco_image_id': None,
            'vis_gen_image_id': None,
            'coco_url': None,
            'vis_gen_url': None,
            'coco_height': None,
            'coco_width': None,
            'vis_gen_height': None,
            'vis_gen_width': None,
            'question_id_bbox': None,
            'file_name': None,
            'split': None
        }
        output['annotations'][str(img)] = annt_stats

    # Write to a file
    with open(output_file, 'w') as fp:
        json.dump(output, fp)


def save_coco_image_annotations(output_file):
    """
    Copy the MS-COCO image annotations from panoptic annotations
    :param output_file: VQD annotation file path
    :return: None
    """
    coco_images = json.load(open('dataset/panoptic_images.json'))['images']
    vqd = json.load(open(output_file))
    vqd_annotations = vqd['annotations']

    for coco_image_id, annt_stats in vqd_annotations.items():
        coco_stats = coco_images[coco_image_id]
        annt_stats['coco_image_id'] = coco_image_id
        annt_stats['coco_url'] = coco_stats['coco_url']
        annt_stats['coco_width'] = coco_stats['width']
        annt_stats['coco_height'] = coco_stats['height']
        annt_stats['split'] = coco_stats['split']
        annt_stats['file_name'] = coco_stats['file_name']

    # Write to a file
    with open(output_file, 'w') as fp:
        json.dump(vqd, fp)


def save_visual_genome_coco_annotations(output_file):
    """
    Copy the Visual Genome image annotations
    :param output_file: VQD annotation file path
    :return: None
    """
    vis_gen_images = json.load(open('dataset/vis_image_annt.json'))
    vqd = json.load(open(output_file))
    vqd_annotations = vqd['annotations']

    for image_id in vis_gen_images.keys():
        vis_gen_stats = vis_gen_images[image_id]
        if vis_gen_stats['coco_id'] is not None:
            annt_stats = vqd_annotations[str(vis_gen_stats['coco_id'])]
            annt_stats['vis_gen_image_id'] = str(vis_gen_stats['image_id'])
            annt_stats['vis_gen_url'] = vis_gen_stats['url']
            annt_stats['vis_gen_width'] = vis_gen_stats['width']
            annt_stats['vis_gen_height'] = vis_gen_stats['height']

    # Write to a file
    with open(output_file, 'w') as fp:
        json.dump(vqd, fp)


def write_to_file(coco_id_to_questions_dict, question_type):
    """
    Write the VQD data to a JSON file.
    :param coco_id_to_questions_dict: A dictionary containing mapping of coco
            image id with list of questions and their respective bounding boxes
    :return: None
    """
    output_file = 'dataset/vqd_annotations.json'

    # Check if output file path exist
    exist = os.path.isfile(output_file)

    if not exist:
        create_fresh_file(output_file)
        save_coco_image_annotations(output_file)
        save_visual_genome_coco_annotations(output_file)

    output = json.load(open(output_file))
    annotations = output['annotations']
    for coco_img_id, questions_dict in coco_id_to_questions_dict.items():
        if coco_img_id in annotations:
            annt_stats = annotations[str(coco_img_id)]
            # If no previous question found for coco image id, then assign the current
            # question and bounding boxes as an answer, else just append the new questions
            # to their respective bounding boxes.
            if len(questions_dict) > 0:
                ques_bbox = questions_dict['question_bbox']
                if annt_stats['question_id_bbox'] is None or \
                        len(annt_stats['question_id_bbox']) == 0:
                    annt_stats['question_id_bbox'] = dict()

                for ques, bboxes in ques_bbox.items():
                    annt_stats['question_id_bbox'][ques] = [bboxes, question_type]

    # Write to a file
    with open(output_file, 'w') as fp:
        json.dump(output, fp)
    print("DONE - Generated: " + output_file)
