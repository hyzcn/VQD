import json
import urllib2
import cv2
import spacy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bbox_1(image, url, bbox_list, ques_list, coco_id, dataset_type, idx=0):

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    image = image[..., ::-1]
    ax.imshow(image)
    ques_str = ""
    edge_color = ['blue', 'green', 'red', 'cyan', 'orange', 'yellow', 'purple', 'brown', 'gray', 'olive']

    for i in range(len(ques_list)):
        ques = ques_list[i]
        bbox = bbox_list[i]
        ec = edge_color[i % len(edge_color)]
        ques_str = ques_str + ques + "(" + str(ec) + ")" + "\n"

        for bb_coord in bbox:
            if len(bb_coord) != 0:
                x, y, w, h = bb_coord
                # Create a Rectangle patch
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=ec, facecolor='none', label="aa")

                # Add the patch to the Axes
                ax.add_patch(rect)
                if dataset_type == 'panoptic':
                    plt.text(x, y, "{}".format(ques), color=ec, fontsize=10)

    if dataset_type == 'vqd':
        plt.title(ques_str)
    plt.xlabel("coco_id: " + str(coco_id) + " URL: " + str(url))
    # plt.show()
    fig.savefig("Images/original_" + str(idx) + ".png")
    plt.close("all")


def draw_bbox(url, bbox_list, ques_list, coco_id, dataset_type, idx=0):
    req = urllib2.Request(url)
    try:
        resp = urllib2.urlopen(req)
    except urllib2.URLError as e:
        print(e.reason)
    # resp = urllib2.urlopen(url)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    image = image[..., ::-1]
    ax.imshow(image)
    ques_str = ""
    edge_color = ['blue', 'green', 'red', 'cyan', 'orange', 'yellow', 'purple', 'brown', 'gray', 'olive']

    for i in range(len(ques_list)):
        ques = ques_list[i]
        bbox = bbox_list[i]
        ec = edge_color[i % len(edge_color)]
        ques_str = ques_str + ques + "(" + str(ec) + ")" + "\n"

        for bb_coord in bbox:
            if len(bb_coord) != 0:
                x, y, w, h = bb_coord
                # Create a Rectangle patch
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=ec, facecolor='none', label="aa")

                # Add the patch to the Axes
                ax.add_patch(rect)
                if dataset_type == 'panoptic':
                    plt.text(x, y, "{}".format(ques), color=ec, fontsize=10)

    if dataset_type == 'vqd':
        plt.title(ques_str)
    plt.xlabel("coco_id: " + str(coco_id) + " URL: " + str(url))
    # plt.show()
    fig.savefig("Images/" + str(idx) + ".png")
    plt.close("all")


def coco_image_filename(image_id):
    image_list = []
    footer = str("0" * (12 - len(image_id))) + str(image_id) + ".jpg"
    image_list.append("COCO_train2014_" + footer)
    image_list.append("COCO_val2014_" + footer)
    return image_list

def load_image(image_list):
    filename_1 = "dataset/train2014/" + image_list[0]
    filename_2 = "dataset/val2014/" + image_list[1]
    image_tensor = None

    try:
        image_tensor = cv2.imread(filename_1)
    except:
        try:
            image_tensor = cv2.imread(filename_2)
        except:
            print("FILE NOT FOUND ERROR")
            sys.exit(1)
    return image_tensor

def get_category_id_to_label(catg):
    # filename = 'data/panoptic_train2017.json'
    # catg = json.load(open(filename))['categories']
    category_ids_to_label = {}
    for cat_dict in catg:
        id = cat_dict['id']
        name = cat_dict['name']
        category_ids_to_label[id] = name
    return category_ids_to_label


def get_coco_img_id_to_vqd_id(ann):
    # filename = 'data/vqd_correction.json'
    # ann = json.load(open(filename))['annotations']
    coco_img_id_to_vqd_id = {}

    for i in ann.keys():
        stats = ann[i]
        if 'coco_image_id' in stats and stats['coco_image_id'] is not None:
            coco_img_id_to_vqd_id[stats['coco_image_id']] = i
    return coco_img_id_to_vqd_id


def get_categories_id_to_bboxes_coord(segments_info):
    categories_id_to_bboxes_coord = {}
    for i in segments_info:
        if i['category_id'] in categories_id_to_bboxes_coord:
            categories_id_to_bboxes_coord[i['category_id']].append(i['bbox'])
        else:
            categories_id_to_bboxes_coord[i['category_id']] = [i['bbox']]
    return categories_id_to_bboxes_coord


def get_categories_name_to_bboxes_coord(segments_info, category_id_to_label):
    categories_name_to_bboxes_coord = {}
    for i in segments_info:
        name = category_id_to_label[i['category_id']]
        if name in categories_name_to_bboxes_coord:
            categories_name_to_bboxes_coord[name].append(i['bbox'])
        else:
            categories_name_to_bboxes_coord[name] = [i['bbox']]
    return categories_name_to_bboxes_coord


def get_noun(ques):
    nlp = spacy.load('en')
    doc = nlp(ques)
    noun_words = set()
    for token in doc:
        if token.pos_ == 'NOUN':
            noun_words.add(token.text)
        # print("{0}\t{1}".format(token.text, token.pos_))
    return noun_words


def get_url(ann_dict):
    if 'coco_url' in ann_dict:
        return ann_dict['coco_url']
    elif 'vis_url' in ann_dict:
        return ann_dict['vis_url']
    else:
        return "Bad URL"


def get_image_id(ann_dict):
    if 'coco_image_id' in ann_dict:
        return ann_dict['coco_image_id']
    elif 'vis_id' in ann_dict:
        return ann_dict['vis_id']
    else:
        return "Bad IMAGE ID"


def vqd_n_panoptic():
    p_filename_t = 'dataset/panoptic_train2017.json'
    # p_filename_v = 'dataset/panoptic_val2017.json'
    v_filename = 'dataset/vqd_correction.json'

    p_f_t = json.load(open(p_filename_t))
    # p_f_v = json.load(open(p_filename_v))
    v_f = json.load(open(v_filename))

    v_annotations = v_f['annotations']
    v_questions = v_f['questions_ids']

    p_annotations = p_f_t['annotations']
    p_categories = p_f_t['categories']

    coco_img_id_to_vqd_id = get_coco_img_id_to_vqd_id(v_annotations)
    category_id_to_label = get_category_id_to_label(p_categories)
    cnt = 0

    for ann_dict in p_annotations:
        coco_img_id = str(ann_dict['image_id'])
        info = ann_dict['segments_info']
        # categories_id_to_bboxes_coord = get_categories_id_to_bboxes_coord(info)
        categories_name_to_bboxes_coord = get_categories_name_to_bboxes_coord(info, category_id_to_label)

        if int(coco_img_id) in coco_img_id_to_vqd_id:
            vqd_id = coco_img_id_to_vqd_id[int(coco_img_id)]
            stats = v_annotations[vqd_id]
            v_url = get_url(stats)
            ques_list = []
            bboxes_list = []
            for qa_pair in stats['qa']:
                bboxes = qa_pair[0]
                if len(bboxes) != 0:
                    ques = v_questions[str(qa_pair[1])]
                    ques_list.append(ques)
                    bboxes_list.append(bboxes)

            cat_names = []
            p_bboxes_list = []
            for key, value in categories_name_to_bboxes_coord.items():
                cat_names.append(key)
                p_bboxes_list.append(value)

            coco_image_list = coco_image_filename(coco_img_id)
            image_tensor = load_image(coco_image_list)
            if image_tensor is None:
                continue
            # draw_bbox(url=v_url, bbox_list=bboxes_list, ques_list=ques_list,
            #           coco_id=coco_img_id, dataset_type='vqd', idx=cnt)
            # cnt += 1
            # draw_bbox(url=v_url, bbox_list=p_bboxes_list, ques_list=cat_names,
            #           coco_id=coco_img_id, dataset_type='panoptic', idx=cnt)
            # cnt += 1

            draw_bbox_1(image=image_tensor, url=v_url, bbox_list=bboxes_list, ques_list=ques_list,
                      coco_id=coco_img_id, dataset_type='vqd', idx=cnt)
            cnt += 1
            draw_bbox_1(image=image_tensor, url=v_url, bbox_list=p_bboxes_list, ques_list=cat_names,
                      coco_id=coco_img_id, dataset_type='panoptic', idx=cnt)
            cnt += 1

        print(cnt)

def bounding_box_distribution(filename):
    js = json.load(open(filename))
    bbox_per_ques = [0] * 40
    cnt = 0

    annt = js['annotations']
    for img_id in annt.keys():
        stats = annt[img_id]
        for qa_pair in stats['qa']:
            if len(qa_pair[0]) == 0:
                cnt += 1
            bbox_per_ques[len(qa_pair[0])] += 1

    print(cnt)
    print(bbox_per_ques)

if __name__ == '__main__':
    vqd_n_panoptic()
    # print(len(get_category_id_to_label()))
    # print(get_coco_img_id_to_vqd_id())
    # filename = 'dataset/vqd_correction.json'
    # bounding_box_distribution(filename)