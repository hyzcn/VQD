# dataset generation

Visual Query Detection (VQD) is a dataset for object detection in Visual Question Answering. Images and annotations in VQD are taken from MS COCO and Visual Genome. Questions in this dataset can be broadly classified into four categories : 

1. Object Recognition

2. Color Reasoning

3. Positional Reasoning

4. Absurd Questions

***vqd_final.json*** is the actual annotation file. This file is a dictionary with 5 keys(*u'train_image_ids',  u'val_image_ids', u'test_image_ids',  u'questions_ids', , u'annotations'*)

#### 1. train_image_ids

It contains a list of image IDs for train split. It is of type **list\<int\>** . The length of list is **67464**.

```python
>>> qs['train_image_ids'][:10]
[64102, 67130, 24801, 26323, 3510, 47930, 34187, 24244, 67162, 39913] 
```

#### 2. val_image_ids

It contains a list of image IDs for validation split. It is of type **list\<int\>** . The length of list is **22489**.

```python
>>> qs['val_image_ids'][:10]
[77428, 70829, 81895, 75956, 70357, 81304, 88015, 78480, 81115, 78375]
```

#### 3. test_image_ids

It contains a list of image IDs for test split. It is of type **list\<int\>** . The length of list is **22485**.

```python
>>> qs['test_image_ids'][:10]
[101114, 111647, 94259, 110450, 106118, 90999, 90171, 96188, 101399, 94396]
```

#### 4. question_ids

It is a dictionary with a key as question ID and value as a question. Both keys and vales are of type **list\<unicode\>**.

```python
>>> qs['questions_ids']['410997']
u'Which frisbees are orange?'

>>> qs['questions_ids']['573305']
u'Which kitchen items are bottles in the image'
```

#### 5. annotations

It is a dictionary with a keys as list of image ID and values 

##### Keys: 

List of image ID of an images in VQD of type  **list\<unicode\>**.

```python
>>> qs['annotations'].keys()[0:10]
[u'50088', u'99682', u'89370', u'89371', u'89372', u'89373', u'89374', u'89375', u'89376', u'89377']

```

##### Values:

1. **VQD_ID:** Each image is stored in this format in the folder 'VQD_images_new'
2. **qa:** List of question and answers. For example - [ [[22,45,120,220],220457],    [[50,100,175,250], 147584] ]. Each bounding box is in [x,y,w,h] format. Please convert the boxes to [x1,y1,x2,y2] format where x1 = x, y1 = y, x2 = x+w and y2 = y+h. The answer are in terms of **bounding box** and the dataset **doesn't contain any text label.**
3. **width:** Width of an image
4. **height:** Height of an image
5. **coco_url:** URL of the image taken from MS-COCO dataset
6. **coco_image_id:** MS-COCO image id
7. **vis_url:** URL of the image taken from Visual7W dataset
8. **vis_id:** Visual7W image id
9. **split:** Denotes whether the image is in train, validation and test split

**Note**: Not all the values number 5, 6, 7, 8 are presents all the time.

```python
>>> qs['annotations'].values()[0]
{
    u'vis_url': u'https://cs.stanford.edu/people/rak248/VG_100K/2357895.jpg', 
    u'width': 500, 
    u'image_id': 50088, 
    u'height': 333, 
    u'qa': [[[[132, 62, 68, 52]], 144000], [[], 453120]],
    u'VQD_ID': u'VQD_50088', 
    u'vis_id': 2357895, 
    u'split': u'train', 
    u'coco_image_id': None
}

```
