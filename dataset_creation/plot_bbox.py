import json
import numpy as np
import urllib
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_large_num_bbox(file_path):

    ds = json.load(open(file_path))
    bbox_file = 'dataset/plot_bbox.json'
    ann = ds['annotations']
    idx = 0

    result = {}

    for image_id in ann.keys():
        stats = ann[image_id]
        qa_pair = stats['qa']
        for qa in qa_pair:
            if len(qa[0]) >= 19:
                ques_id = str(qa[1])
                ques = ds['questions_ids'][ques_id]
                if 'coco_url' in stats:
                    url = stats['coco_url']
                else:
                    url = stats['vis_url']
                bboxes = qa[0]
                result[idx] = {
                    'ques_id': str(ques_id),
                    'ques': ques,
                    'image_id': str(image_id),
                    'url': url,
                    'bbox': bboxes
                }
                idx += 1

    with open(bbox_file, 'w') as fp:
        json.dump(result, fp)
    print("DONE!")
    return bbox_file

def plot_img_bbox(bbox_file):
    ds = json.load(open(bbox_file))
    for i in ds.keys():
        url = ds[i]['url']
        bbox_coord = ds[i]['bbox']
        ques = ds[i]['ques']
        draw_bbox(url, bbox_coord, ques)


def draw_bbox(url, bbox_coord, ques, idx):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    image = image[..., ::-1]
    ax.imshow(image)

    for bb_coord in bbox_coord:
        if len(bb_coord) != 0:
            x, y, w, h = bb_coord
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.title(ques)
    # plt.show()
    fig.savefig("images/big_" + str(idx) + ".png")



def images_with_bboxes(file_path):
    fp = json.load(open(file_path))
    q_dict = fp['questions_ids']
    ann = fp['annotations']
    idx = 1

    for image_id in ann.keys():
        stats = ann[image_id]
        if 'coco_url' in stats:
            url = stats['coco_url']
        else:
            url = stats['vis_url']

        for qa_pair in stats['qa']:
            bbox_coord = qa_pair[0]
            if len(bbox_coord) > 15:
                ques = q_dict[str(qa_pair[1])]
                draw_bbox(url, bbox_coord, ques, idx)
                idx += 1



def save_original_bbox_file():
    orig_file = 'dataset/vqd_final.json'
    bbox_file = 'dataset/plot_bbox.json'

    of = json.load(open(orig_file))
    bb = json.load(open(bbox_file))

    image_list = []
    for i in bb.keys():
        image_list.append(bb[i]['image_id'])

    print(image_list)


if __name__ == '__main__':
    file_path = 'dataset/vqd_correction.json'
    # bbox_file = save_large_num_bbox(file_path)
    # bbox_file = 'dataset/plot_bbox.json'
    # plot_img_bbox(bbox_file)
    # save_original_bbox_file()
    images_with_bboxes(file_path)
