import json
import random
from anytree import Node, RenderTree, find_by_attr


def generate_tree(filename):
    """
    It generates a tree level structure of a MS-COCO things
    and stuff label with leaf node act as an actual label
    while the upper nodes are supercategories
    :param filename: A panoptic categories filename generated
                     from `create_category_json()`
    :return: root node of things and stuff label
    """
    categories = json.load(open(filename))

    things_tree = categories['things']['tree']
    stuff_tree = categories['stuff']['tree']

    # CHILD:PARENT structure
    child_t, _ = things_tree[1].strip().split(':')
    root_things = Node(child_t)
    child_s, _ = stuff_tree[1].strip().split(':')
    root_stuff = Node(child_s)

    for line in things_tree[2:]:
        child, parent = line.strip().split(':')
        Node(child, parent=find_by_attr(root_things, parent))

    for line in stuff_tree[2:]:
        child, parent = line.strip().split(':')
        Node(child, parent=find_by_attr(root_stuff, parent))

    return root_things, root_stuff


def get_same_category_neighbor(name, things_tree):
    """
    Retrieve the sibling name under the same parent.
    e.g: gives `cat` if name=dog since both fall under `animal`
    supercategory
    :param name: A label name
    :param things_tree: A root node of MS-COCO things label
    :return: Sibling name
    """
    for pre, fill, node in RenderTree(things_tree):
        if node.name == name and len(node.children) == 0:
            parent_node = node.parent
            break

    names = []

    # Just for a single child e.g. person category
    if len(parent_node.children) == 1:
        return parent_node.children[0].name

    for node in parent_node.children:
        if node.name != name:
            names.append(node.name)

    return random.choice(names)


def get_different_category_neighbor(name, things_tree):
    """
    Retrieve the sibling name from different parent.
    e.g.: giver `bicycle` if name=dog because both are of
    different supercategory.
    :param name: A label name
    :param things_tree: A root node of MS-COCO things label
    :return: Sibling name of different parent
    """
    for pre, fill, node in RenderTree(things_tree):
        if node.name == name:
            parent_node = node.parent
            superparent_node = node.parent.parent
            break

    supercategories_names = []

    for node in superparent_node.children:
        if node.name != parent_node.name:
            supercategories_names.append(node)

    supercategory = random.choice(supercategories_names)

    names = []
    for node in supercategory.children:
        names.append(node.name)

    return random.choice(names)
