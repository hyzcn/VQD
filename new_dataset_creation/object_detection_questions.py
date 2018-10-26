import json
import random
from utils import *


class ObjectDetectionQues:
    """
    Generates an object detection questions
    """

    def __init__(self):
        self.prefix_type1 = ["Show the", "Show me the"]
        self.prefix_type2 = ["Where is the", "Where are the"]
        self.prefix_type3 = ["Is there a", "Is there any"]
        self.suffix = [None, "in the image", "in the picture"]
        self.plural = load_plural()

    def ques_and_bbox(self, annotations):
        """
        Generates a mapping of questions to bounding boxes for all MS-COCO
        images
        :param annotations: Panoptic annotations
        :return: dict of questions to bounding boxes
        """
        coco_id_ques_dict = {}
        panop_catg_file_p = '../dataset/panoptic_categories.json'
        coco_labels = json.load(open(panop_catg_file_p))['things']['label']

        for coco_img_id, stats in annotations.items():
            catg_name_to_bbox = get_catg_name_to_bbox(stats, coco_labels)
            if len(catg_name_to_bbox) == 0:
                questions_dict = dict()
            else:
                questions_dict = self.ques_to_bboxes_per_image(catg_name_to_bbox)

            coco_id_ques_dict[coco_img_id] = questions_dict
        return coco_id_ques_dict

    def ques_to_bboxes_per_image(self, catg_name_to_bbox):
        """
        Generates a mapping of questions to bounding boxes for a single
        MS-COCO image
        :param catg_name_to_bbox: mapping of category name to bounding boxes
        :return: questions to bounding boxes mapping
        """
        all_ques_to_bboxes_per_image = {}
        for name, bbox in catg_name_to_bbox.items():
            if len(bbox) > 1:
                # Get the plural name if more than one bounding boxes
                name = self.plural[name]
                prefix_type2 = self.prefix_type2[1]
                prefix_type3 = self.prefix_type3[1]

            else:
                prefix_type2 = self.prefix_type2[0]
                prefix_type3 = self.prefix_type3[0]
            prefix_type1 = random.choice(self.prefix_type1)
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

            all_ques_to_bboxes_per_image[question] = bbox

        # limit the number of questions per image
        limit_quest_dict_per_image = {}
        sort_seq = random.choice([True, False])
        num_of_ques = 2
        for k in sorted(all_ques_to_bboxes_per_image,
                        key=lambda k: len(all_ques_to_bboxes_per_image[k]),
                        reverse=sort_seq):
            if num_of_ques > 0:
                limit_quest_dict_per_image[k] = all_ques_to_bboxes_per_image[k]
                num_of_ques -= 1
        return limit_quest_dict_per_image


def main():
    """
    It generates the object detection questions and stores into a VQD
    json file
    :return: None
    """
    panop_ann_file_p = 'drive/My Drive/VQD/dataset/panoptic_annotations.json'
    annotations = json.load(open(panop_ann_file_p))['annotations']
    odq = ObjectDetectionQues()
    coco_id_to_ques_dict = odq.ques_and_bbox(annotations)
    write_to_file(coco_id_to_ques_dict)


if __name__ == '__main__':
    main()
