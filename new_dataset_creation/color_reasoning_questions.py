import re
import random
from utils import *
from object_names_vis_genome import *


class ColorReasoningQues:
    """
    Generates a Color Reasoning Questions
    """

    def __init__(self):
        """
        Constructor to initializes the parameters
        """
        self.prefix = ["Which", "Show me the"]
        self.middle_type1 = ["which is", "which are"]
        self.middle_type2 = ["is", "are"]
        self.suffix = ["in color"]
        self.eos = ['', '?']
        self.color = self.get_color()  # set of color names

    def get_color(self):
        """
        It gives a set of color names
        :return: set of color
        """
        color = {'red', 'green', 'blue', 'white', 'black', 'yellow',
                 'grey', 'orange', 'purple', 'violet', 'pink', 'magenta',
                 'brown', 'silver', 'gold', 'navy'}
        return color

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
                if not bb_iou(new_bbox, bbox, 0.5):
                    return False
        return True

    def ques_to_bboxes_per_image(self, obj_color_keywords_to_bboxes):
        """
        It forms a questions based on object and color word and then
        it generates a set of questions to bounding boxes
        :param obj_color_keywords_to_bboxes: Object-color word to bounding boxes
        :return: Dictionary of questions to bounding boxes
        """
        all_ques_to_bboxes_per_image = dict()
        for keywords, bboxes in obj_color_keywords_to_bboxes.items():
            # split the objects name and color name
            obj_name, color_name = keywords.rsplit(" ", 1)
            prefix = random.choice(self.prefix)
            if prefix.startswith("Show"):
                middle = self.middle_type1[0]
                eos = self.eos[0]
            else:
                middle = self.middle_type2[0]
                eos = self.eos[1]

            # Construct a question
            question = prefix + ' ' + obj_name + ' ' + middle + ' ' \
                       + color_name + ' ' + self.suffix[0] + eos
            all_ques_to_bboxes_per_image[question] = bboxes
        return all_ques_to_bboxes_per_image

    def ques_and_bbox(self, attrib_list, num_ques_per_image):
        """
        It generates a tuple of coco images which is a part of visual genome and
        the rest of visual genome dataset containing set of question to bounding boxes.
        Steps:
            1. Iterate through every visual genome image annotations
            2. Get the object names which is present in coco labels and color names
            3. Form the pair of (object_names, color_names) to bounding boxes
            4. Convert pairs of (object_names, color_names) to questions
            5. Limit the (question, bounding boxes) pair to variable `num_of_ques` with
               decreasing order of maximum number of bounding boxes per question
            6. store the limited (question, bounding boxes) pair
            7. Jump to step-1 and continue till the end

        It also transform the coordinates of visual genome bounding boxes into a
        MS-COCO bounding boxes
        :param attrib_list: A visual genome attributes annotations
        :param num_ques_per_image: Maximum number of questions per image
        :return: tuple of coco and visual genome annotations
        """
        coco_id_ques_bbox = dict()
        vis_id_ques_bbox = dict()
        coco_labels = get_coco_labels()
        vis_image_annt_dict = json.load(open('../dataset/vis_image_annt.json'))
        predefined_objects = set(list(coco_labels) + list(freq_obj_names))

        for attr_dict in attrib_list:
            vis_image_id = attr_dict['image_id']
            attrib = attr_dict['attributes']
            obj_color_keywords_to_bboxes = dict()
            all_ques_to_bboxes_per_image = dict()
            num_of_ques = num_ques_per_image
            for a in attrib:
                if 'names' in a and 'attributes' in a:
                    obj_names = a['names']
                    attr_names = a['attributes']
                    synsets = a['synsets']
                    sent = None
                    attr_names = [re.sub('\W+', '', attr).lower() for attr in attr_names]
                    if len(synsets) == 1:
                        for attr in attr_names:
                            if attr in self.color and len(obj_names) >= 1 and \
                                    obj_names[0] in predefined_objects:
                                sent = obj_names[0] + ' ' + attr
                                break
                    elif len(synsets) > 1:
                        for attr in attr_names:
                            if attr in self.color and len(obj_names) >= 1:
                                obj_name = obj_names[0]
                                for color in self.color:
                                    if color in obj_name:
                                        attr = ''
                                sent = obj_name + ' ' + attr
                                break

                    if sent is not None:
                        # if another (obj, attr) pair sentences finds in annotations then there
                        # is more than one object present of same category
                        if sent in obj_color_keywords_to_bboxes:
                            x, y, w, h = a['x'], a['y'], a['w'], a['h']
                            existing_bboxes = obj_color_keywords_to_bboxes[sent]
                            to_add = True
                            for bbox in existing_bboxes:
                                if not bb_iou(bbox, [x, y, w, h], 0.5):
                                    to_add = False

                            if to_add:
                                obj_color_keywords_to_bboxes[sent].append([x, y, w, h])
                        else:
                            x, y, w, h = a['x'], a['y'], a['w'], a['h']
                            if self.check_redundant_bbox([x, y, w, h], obj_color_keywords_to_bboxes):
                                obj_color_keywords_to_bboxes[sent] = [[x, y, w, h]]

            # Transform the (object, color) name pair to a question
            all_ques_to_bboxes_per_image = self.ques_to_bboxes_per_image(obj_color_keywords_to_bboxes)

            # limit the number of questions per image
            limit_quest_bbox_per_image = dict()
            for k in sorted(all_ques_to_bboxes_per_image, key=lambda k: len(all_ques_to_bboxes_per_image[k]),
                            reverse=True):
                if num_of_ques > 0:
                    limit_quest_bbox_per_image[k] = all_ques_to_bboxes_per_image[k]
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


def main():
    """
    It generates the color reasoning questions and stores into a VQD
    json file
    :return: None
    """
    attrib_filename = '../dataset/attributes.json'
    attrib_list = json.load(open(attrib_filename))
    num_ques_per_image = 2
    crq = ColorReasoningQues()
    coco_id_ques_bbox, vis_id_ques_bbox = crq.ques_and_bbox(attrib_list, num_ques_per_image)
    write_to_file(coco_id_ques_bbox, 'color')


if __name__ == '__main__':
    main()
