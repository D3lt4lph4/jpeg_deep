""" Helper functions for the generation of data-augmented images.

All the functions were taken from https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
"""

import numpy as np

from typing import List

import os

from bs4 import BeautifulSoup


class ConvertTo3Channels:
    '''
    Converts 1-channel and 4-channel images to 3-channel images. Does nothing to images that
    already have 3 channels. In the case of 4-channel images, the fourth channel will be
    discarded.
    '''

    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        if labels is None:
            return image
        else:
            return image, labels


def parse_xml_voc(data_path: str, classes: List[str] = None, exclude_difficult=False):
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
    flagged_boxes = []

    try:
        with open(data_path) as f:
            soup = BeautifulSoup(f, 'xml')
    except FileNotFoundError:
        print("Assuming test mode. Returning dummy boxes for labels")
        boxes.append([0, 0, 0, 0, 0])
        flagged_boxes.append(False)
        return boxes, flagged_boxes

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

        if difficult == 1 and exclude_difficult:
            continue

        # Get the bounding box coordinates.
        bndbox = obj.find('bndbox', recursive=False)
        xmin = int(bndbox.xmin.text)
        ymin = int(bndbox.ymin.text)
        xmax = int(bndbox.xmax.text)
        ymax = int(bndbox.ymax.text)

        boxes.append([class_id, xmin, ymin, xmax, ymax])
        if difficult == 1:
            flagged_boxes.append(True)
        else:
            flagged_boxes.append(False)

    return boxes, flagged_boxes
