import sys

from os.path import join

from copy import deepcopy
from io import BytesIO
from typing import List
from random import shuffle

import numpy as np

from PIL import Image

from tqdm import tqdm

from jpeg2dct.numpy import load, loads

from keras.applications.vgg16 import preprocess_input

from template_keras.generators import TemplateGenerator

from .helper_ssd import BoxFilter
from .helper import parse_xml_voc


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


class VOCGenerator(TemplateGenerator):

    def __init__(self,
                 images_path: List[str],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 label_encoder: object = None,
                 transforms: List[object] = None,
                 dct: bool = False,
                 split_cbcr=False,
                 only_y=False,
                 train_mode: bool = True,
                 labels_output_format: List[str] = (
                     'class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        '''
        # Arguments:
            - images_path: A Python list/tuple or a string representing  a filepath.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': 0,
                              'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}

        self.images = []
        self.images_path = []
        self.transforms = transforms
        self.label_encoder = label_encoder

        for image_dir, set_file in images_path:
            with open(set_file, "r") as f:
                files = [file.rstrip() for file in f.readlines()]

            for filename in files:
                self.images_path.append(
                    join(image_dir, filename + ".jpg"))

        self.images_path = self.images_path
        self.split_cbcr = split_cbcr
        self.only_y = only_y
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._train_mode = train_mode
        self._number_of_data_samples = len(self.images_path)

        self.batch_per_epoch = len(self.images_path) // self._batch_size
        self.indexes = np.arange(len(self.images_path))

        self.labels = []
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
        if index == 0:
            self.on_epoch_end()
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

        # Load the images and labels
        for i in indexes:
            with Image.open(self.images_path[i]) as image:
                image = image.convert("RGB")
                batch_X.append(np.array(image, dtype=np.uint8))
            if self._train_mode:
                batch_y.append(deepcopy(self.labels[i]))
            else:
                batch_y.append([[0, 0, 0, 0, 0]])

        # In case we need to remove any images from the batch, store their indices in this list.
        batch_items_to_remove = []
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
                        continue

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            if self._train_mode and (np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0)):
                batch_y[i] = self.box_filter(batch_y[i])

        for to_remove in batch_items_to_remove[::-1]:
            batch_X.pop(i)
            batch_y.pop(i)

        if self.label_encoder and self._train_mode:
            batch_y_encoded = self.label_encoder(batch_y)
        else:
            batch_y_encoded = batch_y

        if not self.dct:
            for i in range(len(batch_X)):
                batch_X[i] = preprocess_input(batch_X[i])

            batch_X = np.array(batch_X)
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

                temp_y = np.zeros((cb_x * 2, cb_y * 2, y_c), dtype=np.int16)

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

    def prepare_dataset(self, exclude_difficult=False):
        """ We load all the labels when preparing the data. If there is the load in memory option activated, we pre-load the images as well. """

        if not self._train_mode:
            print("Skipping the loading of the parameters as we are in test mode.")
            return None

        it = tqdm(self.images_path,
                  desc='Preparing the dataset', file=sys.stdout)
        for filename in it:

            boxes, flagged_boxes = parse_xml_voc(filename.replace(
                "JPEGImages", "Annotations").replace("jpg", "xml"), exclude_difficult=exclude_difficult)
            self.labels.append(boxes)
            self.flagged_boxes.append(flagged_boxes)

    def get_raw_input_label(self, index):
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

    def get_batch_data(self, index):
        """ Should return the data associated for the batch specified if any. Should return None else.

        # Argument:
            - index: The index of the batch
        """
        batch_X, _ = self.__getitem__(index)
        return batch_X
