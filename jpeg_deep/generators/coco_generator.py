from os.path import basename, join
from copy import deepcopy
import inspect
import sys

import os

from typing import List

from random import shuffle

import numpy as np

from PIL import Image

from tqdm import tqdm, trange

from io import BytesIO

from jpeg2dct.numpy import load, loads

import h5py

from bs4 import BeautifulSoup

from keras.applications.vgg16 import preprocess_input

from .helper_ssd import BoxFilter

from pycocotools.coco import COCO


class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass


class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass


class COCOGenerator(object):

    def __init__(self,
                 image_directory: str,
                 annotation_file: str,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 label_encoder: object = None,
                 transforms: List[object] = None,
                 dct: bool = False,
                 mode: str = "train",
                 split_cbcr: bool=False,
                 only_y: bool=False,
                 labels_output_format: List[str] = (
                     'class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        '''
        Generator for the MS-COCO dataset
        
        # Arguments:
            - image_directory: Directory containing all the images to be used by the generator.
            - annotation_file: File containing all the annotation for the target set.
            - batch_size: The size of the batch to be return by the generator.
            - shuffle: If the images should be shuffle on epoch end.
            - label_encoder: Object to encode the label in the SSD format.
            - transforms: The transformations to apply to the images.
            - dct: If the generator should return the DCT encoded images.
            - mode: The mode of the generator, `train` or `test`, in test mode, the generator will ignore the labels.
            - split_cbcr: If the Cb and Cr inputs should be split in two arrays.
            - only_y: If only the Y component should be returned.
            - labels_output_format: The format of the output (leave as is).
        '''
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': 0,
                              'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}

        self.images_path = []
        self.labels = []

        self.transforms = transforms
        self.label_encoder = label_encoder
        self.split_cbcr = split_cbcr
        self.only_y = only_y

        # Getting all the images
        self.image_directory = image_directory

        self.coco = COCO(annotation_file)

        # display COCO categories and supercategories
        cats = self.coco.loadCats(self.coco.getCatIds())
        id_classes = sorted([(value["id"], value["name"])
                             for value in cats], key=lambda x: x[0])

        # add background class
        id_classes.insert(0, (0, "background"))

        # add the index for the predictions
        id_classes = [(value[0], value[1], i)
                      for i, value in enumerate(id_classes)]

        # create a dictionnary with the ids as keys
        self.matching_dictionnary = {value[0]: [
            value[1], value[2]] for value in id_classes}

        # Loading all the images
        img_ids = self.coco.getImgIds()
        images = self.coco.loadImgs(img_ids)
        # Getting all the information for each of the images
        for image in tqdm(images):
            # Annotations for the image
            file_path = join(image_directory, image["file_name"])

            annIds = self.coco.getAnnIds(imgIds=image['id'], iscrowd=None)
            annotations = self.coco.loadAnns(annIds)
            bounding_boxes = []
            for annotation in annotations:
                bbox = annotation["bbox"]
                bbox = [bbox[0], bbox[1], bbox[0] +
                        bbox[2], bbox[1] + bbox[3]]
                class_id = self.matching_dictionnary[annotation["category_id"]][1]
                bounding_boxes.append([class_id, *bbox])

            if len(bounding_boxes) == 0 and mode == "train":
                continue

            self.images_path.append(file_path)
            self.labels.append(bounding_boxes)

        self.dataset_size = len(self.images_path)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._mode = mode
        self._number_of_data_samples = len(self.images_path)

        self.batch_per_epoch = len(self.images_path) // self._batch_size
        self.indexes = np.arange(len(self.images_path))

        self.image_ids = None
        self.flagged_boxes = []
        self.dct = dct

        self.box_filter = BoxFilter(check_overlap=False, check_min_area=False,
                                    check_degenerate=True, labels_format=self.labels_format)

        self.on_epoch_end()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self.batch_per_epoch = len(self.images_path) // self._batch_size

    @property
    def number_of_data_samples(self):
        return self._number_of_data_samples

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value
        if self.shuffle == False:
            self.indexes = np.arange(len(self.indexes))
        else:
            self.on_epoch_end()

    def __len__(self):
        """ Should return the number of batch per epoch."""
        return self.batch_per_epoch

    def __getitem__(self, index):
        """ Should return the batch target of the index specified.

        # Argument:
            - index: The index of the batch
        """
        index = index % self.batch_per_epoch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self._batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            if self.transforms is not None:
                for transform in self.transforms:
                    transform.labels_format = self.labels_format

        batch_X, batch_y = [], []

        for i in indexes:
            with Image.open(self.images_path[i]) as image:
                image = image.convert("RGB")
                batch_X.append(np.array(image, dtype=np.uint8))
            if not self._mode == "test":
                batch_y.append(deepcopy(self.labels[i]))
            else:
                batch_y.append([[0, 0, 0, 0, 0]])

        # In case we need to remove any images from the batch, store their indices in this list.
        batch_items_to_remove = []
        batch_inverse_transforms = []
        for i in range(len(batch_X)):

            batch_y[i] = np.array(batch_y[i])

            # Apply any image transformations we may have received.
            if self.transforms:
                for transform in self.transforms:
                    batch_X[i], batch_y[i] = transform(
                        batch_X[i], batch_y[i])

                    # In case the transform failed to produce an output image, which is possible for some random transforms.
                    if batch_X[i] is None:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            if (not self._mode == "test") and (np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0)):
                batch_y[i] = self.box_filter(batch_y[i])

        if self.label_encoder and not self._mode == "test":
            batch_y_encoded = self.label_encoder(batch_y)
        else:
            batch_y_encoded = batch_y

        if not self.dct:
            # for i in range(len(batch_X)):
            #     batch_X[i] = preprocess_input(batch_X[i])

            batch_X = np.array(batch_X)
            if len(batch_items_to_remove) > 0:
                print(len(batch_items_to_remove))
            return batch_X, batch_y_encoded
        else:
            X_y = []
            if self.split_cbcr:
                X_cb = []
                X_cr = []
            else:
                X_cbcr = []
            for i, image_to_save in enumerate(batch_X):
                im = Image.fromarray(image_to_save)
                fake_file = BytesIO()
                im.save(fake_file, format="jpeg")

                dct_y, dct_cb, dct_cr = loads(fake_file.getvalue())

                y_x, y_y, y_c = dct_y.shape
                cb_x, cb_y, cb_c = dct_cb.shape

                temp_y = np.zeros((cb_x * 2, cb_y * 2, y_c))

                temp_y[:y_x, :y_y, :] = dct_y

                X_y.append(temp_y)

                if self.split_cbcr:
                    X_cb.append(dct_cb)
                    X_cr.append(dct_cr)
                else:
                    X_cbcr.append(np.concatenate([dct_cb, dct_cr], axis=-1))

            if self.split_cbcr:
                return [np.array(X_y), np.array(X_cb), np.array(X_cr)], batch_y_encoded
            else:
                if self.only_y:
                    return np.array(X_y), batch_y_encoded
                else:
                    return [np.array(X_y), np.array(X_cbcr)], batch_y_encoded

    def get_raw_input_label(self, index:int):
        """ Should return the raw input at a given batch index, i.e something displayable.

        # Argument:
            - index: The index of the batch
        """
        index = index % self.batch_per_epoch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self._batch_size]

        if not (self.labels is None):
            if self.transforms:
                for transform in self.transforms:
                    transform.labels_format = self.labels_format

        batch_X, batch_y = [], []

        for i in indexes:
            with Image.open(self.images_path[i]) as image:
                batch_X.append(np.array(image, dtype=np.uint8))
            batch_y.append(deepcopy(self.labels[i]))

        # In case we need to remove any images from the batch, store their indices in this list.
        batch_items_to_remove = []
        batch_inverse_transforms = []

        for i in range(len(batch_X)):

            batch_y[i] = np.array(batch_y[i])

            # Apply any image transformations we may have received.
            if self.transforms:
                for transform in self.transforms:

                    batch_X[i], batch_y[i] = transform(
                        batch_X[i], batch_y[i])

                    # In case the transform failed to produce an output image, which is possible for some random transforms.
                    if batch_X[i] is None:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):
                batch_y[i] = box_filter(batch_y[i])

        batch_X = np.array(batch_X)
        if self.label_encoder:
            batch_y_encoded = self.label_encoder(batch_y)
        else:
            batch_y_encoded = batch_y

        return batch_X, batch_y_encoded

    def get_batch_data(self, index:int):
        """ Should return the data associated for the batch specified if any. Should return None else.

        # Argument:
            - index: The index of the batch
        """
        batch_X, _ = self.__getitem__(index)
        return batch_X

    def shuffle_batches(self):
        """ Should shuffle the batches of data."""
        shuffle(self.indexes)
