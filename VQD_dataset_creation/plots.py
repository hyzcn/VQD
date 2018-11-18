import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(data, filename, xlabel=None, ylabel=None, title=None):
    """
    Plot the bar chart
    :param data: Array of data
    :param filename: Image filepath to save images
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: Title of plot
    :return: None
    """
    sns.set()
    sns.set_context("paper")
    plt.figure(figsize=(8, 4))
    ticks = [k for k in range(len(data))]
    datanp = np.array(data)
    L = len(datanp)
    c = 'b'
    # To display the number on bar chart
    for i, v in enumerate(data):
        plt.text(x=i, y=v + 5, s=str(v), color='cadetblue',
                 ha='center', va='bottom', fontsize=7)

    plt.bar(range(L), datanp, color=c)
    plt.xticks(range(L), ticks)
    plt.xlim([-0.54, L])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(axis='x', colors='gray')
    plt.tick_params(axis='y', colors='gray')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def length_of_question(t_annot, t_ques_id, v_annot, v_ques_id):
    """
    Plot the bar chart of question length vs. number of questions
    :param t_annot: Train VQD annotations
    :param t_ques_id: Train questions
    :param v_annot: Val VQD annotations
    :param v_ques_id: Val questions
    :return: None
    """
    result = [0] * 15

    for annt in t_annot:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            ques_dict = t_ques_id[int(ques_id)]
            ques = ques_dict['question']
            ques_list = ques.split(" ")
            result[len(ques_list)] += 1
    for annt in v_annot:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            ques_dict = v_ques_id[int(ques_id)]
            ques = ques_dict['question']
            ques_list = ques.split(" ")
            result[len(ques_list)] += 1
    plot_distribution(result, 'dataset/len_question.png',
                      xlabel='Length of questions',
                      ylabel='Number of questions',
                      title='Length of question distribution')


def bounding_box_distribution(t_annot, type=None):
    """
    Plot the bar chart of bounding box distribution
    :param t_annot: Train VQD annotations
    :param type: Train or Val
    :return: None
    """
    result = [0] * 16
    for annt in t_annot:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            if len(bbox) < len(result):
                if len(bbox) == 1:
                    if len(bbox[0]) == 0:
                        result[len(bbox[0])] += 1
                    else:
                        result[len(bbox)] += 1
                else:
                    result[len(bbox)] += 1
    plot_distribution(result, 'dataset/bbox_distribution_' + type + '.png',
                      xlabel='Bounding boxes per questions',
                      ylabel='Number of questions',
                      title='Bounding Box Distribution(' + type + ')')


def question_distribution(t_annot, v_annot):
    """
    Plot the bar chart of question distribution
    :param t_annot: Train VQD annotations
    :param v_annot: Val VQD annotations
    :return: None
    """
    result = [0] * 15
    for annt in t_annot:
        ques_bbox = annt['question_id_bbox']
        result[len(ques_bbox)] += 1
    for annt in v_annot:
        ques_bbox = annt['question_id_bbox']
        result[len(ques_bbox)] += 1
    plot_distribution(result, 'dataset/question_distribution.png',
                      xlabel='Number of questions',
                      ylabel='Number of images',
                      title='Question Distribution')


def question_count_per_category(t_annot, t_ques_id, v_annot, v_ques_id):
    """
    Calculate the total questions per category(simple, color, positional)
    :param t_annot: Train VQD annotations
    :param t_ques_id: Train questions
    :param v_annot: Val VQD annotations
    :param v_ques_id: Val questions
    :return: None
    """
    simple = 0
    positional = 0
    color = 0

    for annt in t_annot:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            ques_dict = t_ques_id[int(ques_id)]
            ques_type = ques_dict['question_type']
            if ques_type == 'simple':
                simple += 1
            elif ques_type == 'color':
                color += 1
            elif ques_type == 'positional':
                positional += 1
            else:
                print("Something is wrong")
    for annt in v_annot:
        ques_bbox = annt['question_id_bbox']
        for ques_id, bbox in ques_bbox.items():
            ques_dict = v_ques_id[int(ques_id)]
            ques_type = ques_dict['question_type']
            if ques_type == 'simple':
                simple += 1
            elif ques_type == 'color':
                color += 1
            elif ques_type == 'positional':
                positional += 1
            else:
                print("Something is wrong")
    print("simple: ", simple)
    print("color: ", color)
    print("positional: ", positional)


if __name__ == '__main__':
    train_f = 'dataset/vqd_train_2014.json'
    val_f = 'dataset/vqd_val_2014.json'
    train_js = json.load(open(train_f))
    val_js = json.load(open(val_f))

    train_annotations = train_js['Annotations']
    train_ques_id = train_js['Question_id']
    val_annotations = val_js['Annotations']
    val_ques_id = val_js['Question_id']

    length_of_question(train_annotations, train_ques_id, val_annotations, val_ques_id)
    bounding_box_distribution(train_annotations, 'Train')
    bounding_box_distribution(val_annotations, 'Val')
    question_distribution(train_annotations, val_annotations)
    question_count_per_category(train_annotations, train_ques_id, val_annotations, val_ques_id)
