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

    def ques_and_bbox(self, catg_name_to_bbox, n):
        """
        Generates a dictionary contains key as questions and bounding boxes
        as values
        :param catg_name_to_bbox: mapping of category name to bounding boxes
        :param n: Number of questions to generate for each coco image
        :return: questions to bounding boxes mapping
        """
        total_questions_dict = {}
        for name, bbox in catg_name_to_bbox.items():
            if len(bbox) > 1:
                name = self.plural[name]
                prefix_type2 = self.prefix_type2[1]
                prefix_type3 = self.prefix_type3[1]

            else:
                prefix_type2 = self.prefix_type2[0]
                prefix_type3 = self.prefix_type3[0]
            prefix_type1 = random.choice(self.prefix_type1)
            prefix_types = {prefix_type1, prefix_type2, prefix_type3}
            questions = []

            for i in range(2):
                prefix = random.sample(prefix_types, 1)[0]
                prefix_types.remove(prefix)
                suffix = random.choice(self.suffix)
                if suffix is None:
                    if prefix.startswith('Show'):
                        questions.append(prefix + ' ' + name)
                    else:
                        questions.append(prefix + ' ' + name + '?')
                else:
                    if prefix.startswith('Show'):
                        questions.append(prefix + ' ' + name + ' ' + suffix)
                    else:
                        questions.append(prefix + ' ' + name + ' ' + suffix + '?')

            total_questions_dict[questions[0]] = bbox
            total_questions_dict[questions[1]] = bbox

        limit_questions_dict = {}
        for i in range(n):
            key = random.choice(total_questions_dict.keys())
            limit_questions_dict[key] = total_questions_dict[key]
            del total_questions_dict[key]
        return limit_questions_dict


def main():
    """
    It generates the object detection questions and stores into a VQD
    json file
    :return: None
    """
    panop_ann_file_p = '../dataset/panoptic_annotations.json'
    panop_catg_file_p = '../dataset/panoptic_categories.json'
    annotations = json.load(open(panop_ann_file_p))['annotations']
    categories = json.load(open(panop_catg_file_p))

    obj = ObjectDetectionQues()
    coco_id_to_ques_dict = {}
    for coco_img_id, stats in annotations.items():
        catg_name_to_bbox = get_catg_name_to_bbox(stats, categories)
        if len(catg_name_to_bbox) == 0:
            questions_dict = dict()
        else:
            questions_dict = obj.ques_and_bbox(catg_name_to_bbox, 2)

        coco_id_to_ques_dict[coco_img_id] = questions_dict

    write_to_file(coco_id_to_ques_dict)


if __name__ == '__main__':
    main()
