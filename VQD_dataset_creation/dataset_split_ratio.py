import json


def change_split(coco_images, vqd_annt, split):
    """
    Change the VQD annotation `split`
    :param coco_images: MS-COCO 2014 image annotations
    :param vqd_annt: VQD annotations
    :param split: 'train' or 'val'
    :return: None
    """
    for image_annt in coco_images:
        image_id = str(image_annt['id'])
        try:
            vqd_annt[image_id]['split'] = split
        except KeyError:
            print('Image-id ' + image_id + 'is not present in the VQD dataset')


def convert_to_2014_coco_split():
    """
    It converts the VQD annotations from MS-COCO 2017 train/val
    split images to MS-COCO 2014 train/val split
    :return: None
    """
    coco_val = 'dataset/annotations/instances_val2014.json'
    coco_train = 'dataset/annotations/instances_train2014.json'
    vqd_fp = 'dataset/vqd_annotations.json'
    val_js = json.load(open(coco_val))
    train_js = json.load(open(coco_train))
    vqd_js = json.load(open(vqd_fp))

    val_images = val_js['images']
    train_images = train_js['images']
    vqd_annt = vqd_js['annotations']

    change_split(train_images, vqd_annt, 'train')
    change_split(val_images, vqd_annt, 'val')

    result = dict()
    result['annotations'] = vqd_annt

    # Write to a file
    with open(vqd_fp, 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    convert_to_2014_coco_split()
