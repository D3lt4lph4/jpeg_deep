""" Helper functions for the generation of data-augmented images.

All the functions were taken from https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
"""

import numpy as np

from typing import List

import os

from bs4 import BeautifulSoup


def parse_xml_voc(data_path: str, classes: List[str] = None):
    '''
    This is an XML parser for datasets in the Pascal VOC format.

    # Arguments:
        - data_path: A list of strings, where each string is the path of a directory that
        - classes (list, optional): A list containing the names of the object classes as found in the
            `name` XML tags. Must include the class `background` as the first list item. The order of this list
            defines the class IDs.
        - exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.

    # Returns:

    '''
    if classes is None:
        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    boxes = []
    eval_difficult = []

    with open(data_path) as f:
        soup = BeautifulSoup(f, 'xml')

    folder = soup.folder.text

    # Get a list of all objects in this image.
    objects = soup.find_all('object')

    # Parse the data for each object.
    for obj in objects:
        class_name = obj.find('name', recursive=False).text
        class_id = classes.index(class_name)
        truncated = int(
            obj.find('truncated', recursive=False).text)
        difficult = int(
            obj.find('difficult', recursive=False).text)

        # Get the bounding box coordinates.
        bndbox = obj.find('bndbox', recursive=False)
        xmin = int(bndbox.xmin.text)
        ymin = int(bndbox.ymin.text)
        xmax = int(bndbox.xmax.text)
        ymax = int(bndbox.ymax.text)

        boxes.append([class_id, xmin, ymin, xmax, ymax])
        if difficult:
            eval_difficult.append(True)
        else:
            eval_difficult.append(False)

    return boxes, eval_difficult
