import sys


def find_noun_simple(ques):
    """
    Extract the noun from `simple` question type category
    :param ques: Question(string of words)
    :return: A noun
    """
    if ques.startswith('Show the'):
        ques = ques.replace('Show the ', '')
        if ' in the image.' in ques:
            ques = ques.replace(' in the image.', '')
        elif ' in the picture.' in ques:
            ques = ques.replace(' in the picture.', '')
        w = ques
    elif ques.startswith('Show me the'):
        ques = ques.replace('Show me the ', '')
        if ' in the image.' in ques:
            ques = ques.replace(' in the image.', '')
        elif ' in the picture.' in ques:
            ques = ques.replace(' in the picture.', '')
        w = ques
    elif ques.startswith('Where is the'):
        ques = ques.replace('Where is the ', '')
        if ' in the image.' in ques:
            ques = ques.replace(' in the image.', '')
        elif ' in the picture.' in ques:
            ques = ques.replace(' in the picture.', '')
        w = ques
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
        ques = ques.replace('Show me the ', '')
        if 'which is ' in ques or 'that is ' in ques:
            ques = ques.replace('which is ', '')
            ques = ques.replace('that is ', '')
        if len(ques.split(' ')) <= 3:
            words = ques.split(' ', 1)
            w = words[1]
        if ' in color.' in ques:
            ques = ques.replace(' in color.', '')
            words = ques.rsplit(' ', 1)
            w = words[0]
    elif ques.startswith('Which '):
        ques = ques.replace('Which ', '')
        ques = ques.replace(' is', '')
        ques = ques.replace(' in color?', '')
        words = ques.rsplit(' ', 1)
        w = words[0]
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
    relationships = {' behind ', ' next to ', ' near ', ' in front of ',
                     ' on top of ', ' under ',
                     ' above ', ' on side of ', ' beside ', ' inside ',
                     ' below ', ' standing next to ',
                     ' to right of ', ' to left of ', ' in back of ',
                     ' behind a '}
    w = ''
    if ques.startswith('Show the'):
        ques = ques.replace('Show the ', '')
        ques = ques.replace(' in the image.', '')
        ques = ques.replace(' in the picture.', '')
    elif ques.startswith('Which '):
        ques = ques.replace('Which ', '')
        ques = ques.replace('is ', '')
    for rel in relationships:
        if rel in ques:
            words = ques.split(rel)
            w = words[0]
            break

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
