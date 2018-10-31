import json
import random
import re
from utils import *
from coco_tree import *


class AbsurdQuestionSimple:
    """
    Generates a simple absurd questions
    """

    def __init__(self):
        """
        Constructor to initializes the parameters
        """
        self.prefix_type1 = ["Show the", "Show me the"]
        self.prefix_type2 = ["Where is the"]
        self.prefix_type3 = ["Is there a"]
        self.suffix = [None, "in the image", "in the picture"]

    def ques_and_bbox(self, annotations):
        """
        Generates a mapping of questions to bounding boxes for all MS-COCO
        images. Here bounding boxes are empty since it is an absurd questions
        :param annotations: Panoptic annotations
        :return: dict of questions to empty bounding boxes
        """
        coco_id_ques_dict = {}
        panop_catg_file_p = '../dataset/panoptic_categories.json'
        coco_labels = json.load(open(panop_catg_file_p))['things']['label']
        things_tree, stuff_tree = generate_tree(panop_catg_file_p)

        for coco_img_id, stats in annotations.items():
            catg_names = get_unique_categs(stats, coco_labels)
            if len(catg_names) == 0:
                questions_dict = dict()
            else:
                questions_dict = self.ques_to_bboxes_per_image(catg_names, things_tree)

            coco_id_ques_dict[coco_img_id] = questions_dict
        return coco_id_ques_dict

    def ques_to_bboxes_per_image(self, catg_names, things_tree):
        """
        Generates a mapping of questions to empty bounding boxes for a single
        MS-COCO image
        :param catg_names: mapping of category name to bounding boxes
        :return: questions to empty bounding boxes mapping
        """
        all_ques_to_bboxes_per_image = dict()
        neighbor_catg_names = set()
        # Get the sibling name of same parent
        for name in catg_names:
            n_names = get_same_category_neighbor(name, things_tree)
            if n_names != name:
                neighbor_catg_names.add(n_names)

        for name in neighbor_catg_names:
            prefix_type1 = random.choice(self.prefix_type1)
            prefix_type2 = self.prefix_type2[0]
            prefix_type3 = self.prefix_type3[0]
            prefix_types = {prefix_type1, prefix_type2, prefix_type3}

            prefix = random.sample(prefix_types, 1)[0]
            suffix = random.choice(self.suffix)
            if suffix is None:
                if prefix.startswith('Show'):
                    question = prefix + ' ' + name
                else:
                    question = prefix + ' ' + name + '?'
            else:
                if prefix.startswith('Show'):
                    question = prefix + ' ' + name + ' ' + suffix
                else:
                    question = prefix + ' ' + name + ' ' + suffix + '?'

            all_ques_to_bboxes_per_image[question] = [[]]

        # limit the number of questions per image
        limit_quest_dict_per_image = {}
        num_of_ques = 2
        for question in all_ques_to_bboxes_per_image:
            if num_of_ques > 0:
                limit_quest_dict_per_image[question] = all_ques_to_bboxes_per_image[question]
                num_of_ques -= 1
        return limit_quest_dict_per_image


class AbsurdQuestionColor:
    """
    Generates a Color Reasoning Absurd Questions
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
        self.color = self.get_color()

    def get_color(self):
        """
        It gives a set of color names
        :return: set of color
        """
        color = {'red', 'green', 'blue', 'white', 'black', 'yellow',
                 'grey', 'orange', 'purple', 'violet', 'pink', 'magenta',
                 'brown', 'silver', 'gold', 'navy'}
        return color

    def random_except(self, color_name):
        """
        Get a different color name from the pre-defined color
        :param color_name: color name
        :return: A different color name
        """
        random_name = color_name
        colors = list(self.color)
        while random_name == color_name:
            random_name = random.choice(colors)
        return random_name

    def ques_to_bboxes_per_image(self, obj_color_keywords_to_bboxes):
        """
        It forms a questions based on object and color word and then
        it generates a set of questions to an empty bounding boxes
        :param obj_color_keywords_to_bboxes: Object-color word to empty bounding boxes
        :return: Dictionary of questions to empty bounding boxes
        """
        all_ques_to_bboxes_per_image = dict()
        for keywords, bboxes in obj_color_keywords_to_bboxes.items():
            obj_name, color_name = keywords.rsplit(" ", 1)

            prefix = random.choice(self.prefix)
            if prefix.startswith("Show"):
                middle = self.middle_type1[0]
                eos = self.eos[0]
            else:
                middle = self.middle_type2[0]
                eos = self.eos[1]

            sentence = prefix + ' ' + obj_name + ' ' + middle + ' ' \
                       + color_name + ' ' + self.suffix[0] + eos
            all_ques_to_bboxes_per_image[sentence] = bboxes
        return all_ques_to_bboxes_per_image

    def ques_and_bbox(self, attrib_list):
        """
        It generates a tuple of coco images which is a part of visual genome and
        the rest of visual genome dataset containing set of question to bounding boxes.
        Steps:
            1. Iterate through every visual genome image annotations
            2. Get the object names which is present in coco labels and color names
            3. Get the different object and color name which is not present in the image
            4. Form the pair of (object_names, color_names) to empty bounding boxes
            5. Convert pairs of (object_names, color_names) to questions
            6. Limit the (question, bounding boxes) pair to variable `num_of_ques`
            7. store the limited (question, bounding boxes) pair
            8. Jump to step-1 and continue till the end

        :param attrib_list: A visual genome attributes annotations
        :return: tuple of coco and visual genome annotations
        """
        coco_id_ques_dict = dict()
        vis_id_ques_dict = dict()
        vis_image_annt_dict = json.load(open('../dataset/vis_image_annt.json'))

        # Generate a MS-COCO things label tree structure for getting a different
        # label
        panop_catg_file_p = '../dataset/panoptic_categories.json'
        things_tree, stuff_tree = generate_tree(panop_catg_file_p)
        categories = json.load(open(panop_catg_file_p))
        coco_labels = categories['things']['label'].values()

        for attr_dict in attrib_list:
            vis_image_id = attr_dict['image_id']
            attrib = attr_dict['attributes']
            obj_color_keywords_to_bboxes = dict()
            for a in attrib:
                if 'names' in a and 'attributes' in a:
                    obj_names = a['names']
                    attr_names = a['attributes']
                    attr_names = [re.sub('\W+', '', attr).lower() for attr in attr_names]
                    for obj in obj_names:
                        for attr in attr_names:
                            if attr in self.color and obj in coco_labels:
                                # Get the different color name and the different label of
                                # same parent
                                attr = self.random_except(attr)
                                obj = get_same_category_neighbor(obj, things_tree)
                                sent = obj + ' ' + attr
                                obj_color_keywords_to_bboxes[sent] = [[]]

            # Transform the (object, color) name pair to a question
            all_ques_to_bboxes_per_image = self.ques_to_bboxes_per_image(obj_color_keywords_to_bboxes)

            # limit the number of questions per image
            num_of_ques = 2
            limit_quest_dict_per_image = dict()
            for question in all_ques_to_bboxes_per_image:
                if num_of_ques > 0:
                    limit_quest_dict_per_image[question] = all_ques_to_bboxes_per_image[question]
                    num_of_ques -= 1

            image_stats = vis_image_annt_dict[str(vis_image_id)]

            # Store the (question, bounding boxes) pair to coco_dict if `coco_id` is present
            # else save it in vis_dict
            if image_stats['coco_id'] is None:
                vis_id_ques_dict[vis_image_id] = limit_quest_dict_per_image
            else:
                coco_id_ques_dict[str(image_stats['coco_id'])] = {'qa': limit_quest_dict_per_image,
                                                                  'vis_height': image_stats['height'],
                                                                  'vis_width': image_stats['width'],
                                                                  'url': image_stats['url']}
        return coco_id_ques_dict, vis_id_ques_dict


def main():
    """
    It generates the object detection questions and stores into a VQD
    json file
    :return: None
    """
    panop_ann_file_p = '../dataset/panoptic_annotations.json'
    annotations = json.load(open(panop_ann_file_p))['annotations']
    aqs = AbsurdQuestionSimple()
    coco_id_to_ques_dict = aqs.ques_and_bbox(annotations)
    write_to_file(coco_id_to_ques_dict)

    visual_genome_attrib_file_p = '../dataset/attributes.json'
    attrib_list = json.load(open(visual_genome_attrib_file_p))
    aqc = AbsurdQuestionColor()
    coco_id_ques_dict, vis_id_ques_dict = aqc.ques_and_bbox(attrib_list)
    write_to_file(coco_id_ques_dict)


if __name__ == '__main__':
    main()
