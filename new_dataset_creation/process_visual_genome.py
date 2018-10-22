import json


def create_image_annot_json():
    """
    The original Visual Genome `image_data.json` file is a list
    of dictionary. This method generates a dictionary with key
    as visual genome image id and value as image stats to find the
    image id in Big(O(1)) time complexity
    :param filepath: A image_data.json file path
    :return: None
    """
    filepath = '../dataset/image_data.json'
    img_list = json.load(open(filepath))
    result = dict()
    for img in img_list:
        vis_id = img['image_id']
        result[vis_id] = img

    dest_file_path = '../dataset/vis_image_annt.json'
    with open(dest_file_path, 'w') as fp:
        json.dump(result, fp)
    print("DONE! - Generated " + dest_file_path)


if __name__ == '__main__':
    create_image_annot_json()
