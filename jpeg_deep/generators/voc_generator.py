from __future__ import division

from os.path import basename, join
from copy import deepcopy
import inspect
import sys

import os

from typing import List

from random import shuffle

import numpy as np

import sklearn.utils

from PIL import Image
import cv2

from tqdm import tqdm, trange

from io import BytesIO

from jpeg2dct.numpy import load, loads

import h5py

from bs4 import BeautifulSoup

from .helper_ssd import SSDInputEncoder
from .helper_ssd import BoxFilter

from .helper import parse_xml_voc

from template_keras.generators import TemplateGenerator


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
                 batch_size: int = 32,
                 shuffle: bool = True,
                 label_encoder: object = None,
                 transforms: List[object] = None,
                 load_images_into_memory: bool = False,
                 images_path: List[str] = None,
                 labels_output_format: List[str] = (
                     'class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        '''
        # Arguments:
            - load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
            - images_path: A Python list/tuple or a string representing  a filepath.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': 0,
                              'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}

        self.load_images_into_memory = load_images_into_memory

        self.images = []
        self.images_path = []
        self.transforms = transforms
        self.label_encoder = label_encoder

        if not images_path is None:
            for file in images_path:
                splitted = file.split("/")
                directory_voc = "/".join(splitted[:6])
                with open(file) as description_file:
                    files = description_file.read().splitlines()

                for filename in files:
                    self.images_path.append(
                        join(directory_voc, "JPEGImages", filename + ".jpg"))

            self.dataset_size = len(self.images_path)
        else:
            self.images_path = None

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._number_of_data_samples = len(self.images_path)

        self.batches_per_epoch = len(self.images_path) // self._batch_size
        self.indexes = np.arange(len(self.images_path))

        self.labels = []
        self.image_ids = None
        self.difficult = None

        self.box_filter = BoxFilter(check_overlap=False, check_min_area=False,
                                    check_degenerate=True, labels_format=self.labels_format)

        self.on_epoch_end()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def number_of_data_samples(self):
        return self._number_of_data_samples

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value

    def __len__(self):
        """ Should return the number of batch per epoch."""
        return self.batch_per_epoch

    def __getitem__(self, index):
        """ Should return the batch target of the index specified.

        # Argument:
            - index: The index of the batch
        """
        index = index % self.batches_per_epoch
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

        if self.load_images_into_memory:
            for i in batch_indices:
                batch_X.append(self.images[i])
                batch_y.append(deepcopy(self.labels[i]))
        else:
            for i in indexes:
                with Image.open(self.images_path[i]) as image:
                    batch_X.append(np.array(image, dtype=np.uint8))
                batch_y.append(deepcopy(self.labels[i]))

        # elif not (self.hdf5_dataset is None):
        #     for i in batch_indices:
        #         batch_X.append(self.hdf5_dataset['images'][i].reshape(
        #             self.hdf5_dataset['image_shapes'][i]))

        # if not (self.eval_neutral is None):
        #     batch_eval_neutral = self.eval_neutral[current:current+batch_size]
        # else:
        #     batch_eval_neutral = None

        # # Get the image IDs for this batch (if there are any).
        # if not (self.image_ids is None):
        #     batch_image_ids = self.image_ids[current:current+batch_size]
        # else:
        #     batch_image_ids = None

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

        X_y = []
        X_cbcr = []
        for i, image_to_save in enumerate(batch_X):
            im = Image.fromarray(image_to_save)
            fake_file = BytesIO()
            im.save(fake_file, format="jpeg")

            dct_y, dct_cb, dct_cr = loads(fake_file.getvalue())

            y_x, y_y, y_c = dct_y.shape
            cb_x, cb_y, cb_c = dct_cb.shape

            temp_y = np.empty((cb_x * 2, cb_y * 2, y_c))

            temp_y[:y_x, :y_y, :] = dct_y

            X_y.append(temp_y)
            X_cbcr.append(np.concatenate([dct_cb, dct_cr], axis=-1))

        return [np.array(X_y), np.array(X_cbcr)], batch_y_encoded

    def prepare_dataset(self):
        """ We load all the labels when preparing the data. If there is the load in memory option activated, we pre-load the images as well. """

        if self.load_images_into_memory:
            print("The images will be loaded into memory.")

        it = tqdm(self.images_path,
                  desc='Preparing the dataset', file=sys.stdout)
        for filename in it:
            if self.load_images_into_memory:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
            boxes, _ = parse_xml_voc(filename.replace(
                "JPEGImages", "Annotations").replace("jpg", "xml"))
            self.labels.append(boxes)

    def get_raw_input_label(self, index):
        """ Should return the raw input at a given batch index, i.e something displayable.

        # Argument:
            - index: The index of the batch
        """
        index = index % self.batches_per_epoch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self._batch_size]

        if not (self.labels is None):
            if self.transforms:
                for transform in self.transforms:
                    transform.labels_format = self.labels_format

        batch_X, batch_y = [], []

        if self.load_images_into_memory:
            for i in batch_indices:
                batch_X.append(self.images[i])
                batch_y.append(deepcopy(self.labels[i]))
        else:
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

    def shuffle_batches(self):
        """ Should shuffle the batches of data."""
        shuffle(self.indexes)

    def load_hdf5_dataset(self, verbose=True):
        '''
        Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
        produces.

        Arguments:
            verbose (bool, optional): If `True`, prints out the progress while loading
                the dataset.

        Returns:
            None.
        '''

        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size,
                            desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(
                    self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size,
                            desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size,
                            desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(
                    self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        '''
        Converts the currently loaded dataset into a HDF5 file. This HDF5 file contains all
        images as uncompressed arrays in a contiguous block of memory, which allows for them
        to be loaded faster. Such an uncompressed dataset, however, may take up considerably
        more space on your hard drive than the sum of the source images in a compressed format
        such as JPG or PNG.

        It is recommended that you always convert the dataset into an HDF5 dataset if you
        have enugh hard drive space since loading from an HDF5 dataset accelerates the data
        generation noticeably.

        Note that you must load a dataset (e.g. via one of the parser methods) before creating
        an HDF5 dataset from it.

        The created HDF5 dataset will remain open upon its creation so that it can be used right
        away.

        Arguments:
            file_path (str, optional): The full file path under which to store the HDF5 dataset.
                You can load this output file via the `DataGenerator` constructor in the future.
            resize (tuple, optional): `False` or a 2-tuple `(height, width)` that represents the
                target size for the images. All images in the dataset will be resized to this
                target size before they will be written to the HDF5 file. If `False`, no resizing
                will be performed.
            variable_image_size (bool, optional): The only purpose of this argument is that its
                value will be stored in the HDF5 dataset in order to be able to quickly find out
                whether the images in the dataset all have the same size or not.
            verbose (bool, optional): Whether or not prit out the progress of the dataset creation.

        Returns:
            None.
        '''

        self.hdf5_dataset_path = file_path

        dataset_size = len(self.filenames)

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(file_path, 'w')

        # Create a few attributes that tell us what this dataset contains.
        # The dataset will obviously always contain images, but maybe it will
        # also contain labels, image IDs, etc.
        hdf5_dataset.attrs.create(
            name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(
            name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(
            name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # It's useful to be able to quickly check whether the images in a dataset all
        # have the same size or not, so add a boolean attribute for that.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(
                name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(
                name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Create the dataset in which the images will be stored as flattened arrays.
        # This allows us, among other things, to store images of variable size.
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Create the dataset that will hold the image heights, widths and channels that
        # we need in order to reconstruct the images from the flattened arrays later.
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(
                                                            dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if not (self.labels is None):

            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # Create the dataset that will hold the dimensions of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(
                                                                dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)

            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if not (self.image_ids is None):

            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))

            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if not (self.eval_neutral is None):

            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(
                                                                dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset',
                        file=sys.stdout)
        else:
            tr = range(dataset_size)

        # Iterate over all images in the dataset.
        for i in tr:

            # Store the image.
            with Image.open(self.filenames[i]) as image:

                image = np.asarray(image, dtype=np.uint8)

                # Make sure all images end up having three channels.
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Flatten the image array and write it to the images dataset.
                hdf5_images[i] = image.reshape(-1)
                # Write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # Store the ground truth if we have any.
            if not (self.labels is None):

                labels = np.asarray(self.labels[i])
                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)
                # Write the labels' shape to the label shapes dataset.
                hdf5_label_shapes[i] = labels.shape

            # Store the image ID if we have one.
            if not (self.image_ids is None):

                hdf5_image_ids[i] = self.image_ids[i]

            # Store the evaluation-neutrality annotations if we have any.
            if not (self.eval_neutral is None):

                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
