import sys


def find_noun_simple(ques):
    """
    Extract the noun from `simple` question type category
    :param ques: Question(string of words)
    :return: A noun
    """
    if ques.startswith('Show the'):
        words = ques.split(' ')
        w = words[2]
    elif ques.startswith('Show me the'):
        words = ques.split(' ')
        w = words[3]
    elif ques.startswith('Where is the'):
        words = ques.split(' ')
        w = words[3]
    else:
        print("problem with simple question")
        return None
    w = w.replace('.', '')
    w = w.replace('?', '')
    return w


def find_noun_color(ques):
    """
    Extract the noun from `color` question type category
    :param ques: Question(string of words)
    :return: A noun
    """
    if ques.startswith('Show me the'):
        words = ques.split(' ')
        if len(words) > 6:
            w = words[3]
        else:
            w = words[4]
    elif ques.startswith('Which'):
        words = ques.split(' ')
        w = words[1]
    else:
        print("problem with color question")
        return None
    w = w.replace('.', '')
    w = w.replace('?', '')
    return w


def find_noun_positional(ques):
    """
    Extract the noun from `positional` question type category
    :param ques: Question(string of words)
    :return: A noun
    """
    relationships = {'behind', 'next to', 'near', 'in front of', 'on top of',
                     'under', 'above', 'on side of', 'beside', 'inside',
                     'below', 'standing next to', 'to right of', 'to left of',
                     'in back of', 'behind a'}
    predicate = None
    ques = ques.lower()
    for rel in relationships:
        if rel in ques:
            predicate = rel
    if predicate is None:
        print('problem in predicate')
        print(ques)
        sys.exit(1)

    my_list = ques.split(predicate)

    if ques.startswith('show the'):
        words = my_list[0].split(' ')
        subject = words[2]
        words = my_list[1].split(' ')
        obj = words[0]
        w = subject
    elif ques.startswith('which'):
        words = my_list[0].split(' ')
        subject = words[1]
        words = my_list[1].split(' ')
        obj = words[0]
        w = subject
    else:
        print("problem with positional question")
        return None
    w = w.replace('.', '')
    w = w.replace('?', '')
    return w


def get_noun(ques, ques_type):
    """
    Get the noun from question based on question types
    :param ques: Question(string of words)
    :param ques_type: A question type(`simple`, `color`, or `positional`)
    :return: A noun
    """
    if ques_type == 'simple':
        return find_noun_simple(ques)
    elif ques_type == 'color':
        return find_noun_color(ques)
    elif ques_type == 'positional':
        return find_noun_positional(ques)
    else:
        print("ERROR: Issue with question or question_type")
        sys.exit(1)


if __name__ == '__main__':
    ques = 'Show the car in the picture.'
    ques_type = 'simple'
    print(get_noun(ques, ques_type))

    ques = 'Which cat is grey in color?'
    ques_type = "color"
    print(get_noun(ques, ques_type))

    ques = 'Show the animal near elephant in the image.'
    ques_type = "positional"
    print(get_noun(ques, ques_type))
