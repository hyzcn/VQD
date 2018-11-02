import json
import random
from utils import *


class PositionReasoningQues:
    """
    Generates a Position Reasoning Questions
    """

    def __init__(self):
        """
        Constructor to initializes the parameters
        """
        self.prefix = ['Which', 'Show the']
        self.middle_type = ['is']
        self.suffix = ['in the picture', 'in the image', '']
        self.eos = ['?']
        self.delimiter = '<=>'
        self.relationships = self.get_relationships()

    @staticmethod
    def get_relationships():
        relationships = {'behind', 'next to', 'near', 'in front of', 'on top of', 'under',
                         'above', 'on side of', 'beside', 'inside', 'below', 'standing next to',
                         'to right of', 'to left of', 'in back of', 'behind a'}
        return relationships

    def check_redundant_bbox(self, new_bbox, obj_color_keywords_to_bboxes):
        """
        Check for redundant bounding box with IOU if same object is present
        with different color attributes
        :param new_bbox: New bounding box to add or not
        :param obj_color_keywords_to_bboxes: Existing bounding boxes
        :return: True if it's unique else False
        """
        for sent, bboxes in obj_color_keywords_to_bboxes.items():
            for bbox in bboxes:
                if not bb_iou(new_bbox, bbox):
                    return False
        return True

    def sent_bbox_to_ques_bbox(self, sent_box):
        """
        It transform the (subject, predicate, object) sentence to a positional
        reasoning type question and map question of their respective bounding
        boxes.
        :param sent_box: A sentence containing (subject, predicate, object) and bounding box
                         separated by delimiter
        :return: A dictionary with mapping of question to bounding boxes
        """
        ques_bbox_dict = dict()
        for sent, bboxes in sent_box.items():
            subj, predicate, obj = sent.split(self.delimiter)

            prefix = random.choice(self.prefix)
            if prefix.startswith("Show"):
                middle = ''
                suffix = random.choice(self.suffix)
                eos = ''
                question = prefix + ' ' + subj + ' ' + predicate + ' ' + obj + ' ' + suffix + eos
            else:
                middle = self.middle_type[0]
                suffix = ''
                eos = self.eos[0]
                question = prefix + ' ' + subj + ' ' + middle + ' ' + predicate + ' ' + obj + eos

            ques_bbox_dict[question] = bboxes
        return ques_bbox_dict

    def get_subj_pred_obj_and_bboxes(self, rel_annt):
        """
        It forms a unique pair of (subject, predicate, object) and bounding box
        separated by delimiter
        :param rel_annt: A single instance relationship dict of an image
        :return: A sentence and bounding box
        """
        sub_name = rel_annt['subject']['name']
        predicate = rel_annt['predicate']
        obj_name = rel_annt['object']['name']
        sent = None
        bbox = None
        if sub_name != obj_name:
            bboxes = [int(rel_annt['subject']['x']), int(rel_annt['subject']['y']),
                      int(rel_annt['subject']['w']), int(rel_annt['subject']['h'])]
            sent = sub_name + self.delimiter + predicate + \
                   self.delimiter + obj_name
            bbox = [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]
        return sent, bbox

    def get_ques_and_bbox(self, relation_list, num_ques_per_image):
        """
        It generates a tuple of coco images which is a part of visual genome and
        the rest of visual genome dataset containing set of question to bounding boxes.
        Steps:
            1. Iterate through every visual genome image annotations
            2. Iterate through every relationship inside that particular image
            3. Form a string of (subject, predicate, object, bounding box) for that
               relationship if the predicate matches with our positional relationships
            4. Transform all the above sentence structure to a (question, bboxes) mapping
            5. Limit the (question, bounding boxes) pair to variable `num_of_ques` with
               decreasing order of maximum number of bounding boxes per question
            6. store the limited (question, bounding boxes) pair
            7. Jump to step-1 and continue till the end of images

        It also transform the coordinates of visual genome bounding boxes into a
        MS-COCO bounding boxes
        :param relation_list: A visual genome attributes annotations
        :param num_ques_per_image: Maximum number of questions per image
        :return: A tuple of coco and visual genome annotations
        """
        coco_id_ques_bbox = dict()
        vis_id_ques_bbox = dict()
        vis_image_annt_dict = json.load(open('../dataset/vis_image_annt.json'))

        for i, rel_dict in enumerate(relation_list):
            sentence_keywords_to_bboxes = dict()
            rel = rel_dict['relationships']
            vis_image_id = rel_dict['image_id']
            num_of_ques = num_ques_per_image

            for rel_annt in rel:
                pred = rel_annt['predicate'].lower()
                # only for positional reasoning relationships
                if pred in self.relationships:
                    sent, bbox = self.get_subj_pred_obj_and_bboxes(rel_annt)
                    if sent is not None:
                        if sent in sentence_keywords_to_bboxes:
                            existing_bboxes = sentence_keywords_to_bboxes[sent]
                            to_add = True
                            for ex_bbox in existing_bboxes:
                                if not bb_iou(ex_bbox, bbox):
                                    to_add = False
                            if to_add:
                                sentence_keywords_to_bboxes[sent].append(bbox)
                        else:
                            if self.check_redundant_bbox(bbox, sentence_keywords_to_bboxes):
                                sentence_keywords_to_bboxes[sent] = [bbox]

            # A sentence of (subject, predicate, object, bounding box) to (question, bounding box)
            ques_bbox_dict_per_image = self.sent_bbox_to_ques_bbox(sentence_keywords_to_bboxes)

            # limit the number of questions per image
            limit_quest_bbox_per_image = dict()
            for k in sorted(ques_bbox_dict_per_image, key=lambda k: len(ques_bbox_dict_per_image[k]),
                            reverse=True):
                if num_of_ques > 0:
                    limit_quest_bbox_per_image[k] = ques_bbox_dict_per_image[k]
                    num_of_ques -= 1

            image_stats = vis_image_annt_dict[str(vis_image_id)]

            # Store the (question, bounding boxes) pair to coco_dict if `coco_id` is present
            # else save it in vis_dict
            if image_stats['coco_id'] is None:
                vis_id_ques_bbox[str(vis_image_id)] = limit_quest_bbox_per_image
            else:
                coco_id_ques_bbox[str(image_stats['coco_id'])] = {'question_bbox': limit_quest_bbox_per_image,
                                                             'vis_height': image_stats['height'],
                                                             'vis_width': image_stats['width'],
                                                             'url': image_stats['url']}
        # Transform the bounding boxes from visual genome image dimension to a
        # MS-COCO image dimension
        coco_id_ques_bbox = transform_vis_bbox_to_coco_bbox(coco_id_ques_bbox)
        return coco_id_ques_bbox, vis_id_ques_bbox


if __name__ == '__main__':
    """
    It generates the positional reasoning questions and stores into a VQD
    json file
    :return: None
    """
    relationship_path = '../dataset/relationships.json'
    relation_list = json.load(open(relationship_path))
    num_ques_per_image = 2
    prq = PositionReasoningQues()
    coco_id_ques_bbox, vis_id_ques_bbox = prq.get_ques_and_bbox(relation_list, num_ques_per_image)
    write_to_file(coco_id_ques_bbox)
