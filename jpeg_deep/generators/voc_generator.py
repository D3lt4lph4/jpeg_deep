from __future__ import division

from os.path import basename, join
from copy import deepcopy
import inspect
import sys

import os

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

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

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


class DataGeneratorDCT(TemplateGenerator):

    def __init__(self,
                 batch_size: int = 32,
                 shuffle: bool = True,
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

        self.images = None
        self.images_path = []

        if not images_path is None:
            for file in images_path:
                with open(file) as description_file:
                    self.images_path += my_file.readlines()

            self.dataset_size = len(self.images_path)

            if load_images_into_memory:
                self.images = []
                it = tqdm(self.images_path,
                          desc='Loading images into memory', file=sys.stdout)
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.images_path = None

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._number_of_data_samples = len(self.images_path)

        self.batch_per_epoch = len(self.images_path) // self._batch_size
        self.indexes = np.arange(len(self.images_path))

        self.labels = None
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
        """ To be called at the end of an epoch. The indexes of the data could be shuffle here."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        batch_X, batch_y = [], []

        if self.load_images_into_memory:
            for i in batch_indices:
                batch_X.append(self.images[i])
                batch_y.append(deepcopy(self.labels[i]))
        else:
            for i in batch_indices:
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
            if transformations:
                for transform in transformations:

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
                if (batch_y[i].size == 0) and not keep_images_without_gt:
                    batch_items_to_remove.append(i)

        # Remove any items we might not want to keep from the batch.
        if batch_items_to_remove:
            for j in sorted(batch_items_to_remove, reverse=True):
                # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                batch_X.pop(j)
                batch_filenames.pop(j)
                if batch_inverse_transforms:
                    batch_inverse_transforms.pop(j)
                if not (self.labels is None):
                    batch_y.pop(j)
                if not (self.image_ids is None):
                    batch_image_ids.pop(j)
                if not (self.eval_neutral is None):
                    batch_eval_neutral.pop(j)
                if 'original_images' in returns:
                    batch_original_images.pop(j)
                if 'original_labels' in returns and not (self.labels is None):
                    batch_original_labels.pop(j)

        batch_X = np.array(batch_X)
        batch_y_encoded = label_encoder(batch_y, diagnostics=False)

        # Compose the output.
        new_batch_X = np.empty(batch_X.shape, dtype=np.int32)

        X_y = np.empty((batch_X.shape[0], 38, 38, 64))
        X_cbcr = np.empty((batch_X.shape[0], 19, 19, 128))
        for i, image_to_save in enumerate(batch_X):
            im = Image.fromarray(image_to_save)
            fake_file = BytesIO()
            im.save(fake_file, format="jpeg")

            dct_y, dct_cb, dct_cr = loads(fake_file.getvalue())

            X_y[i] = dct_y
            X_cbcr[i] = np.concatenate([dct_cb, dct_cr], axis=-1)

        return [X_y, X_cbcr], batch_y_encoded

    def prepare_dataset(self):
        pass

    def get_raw_input_label(self, index):
        """ Should return the raw input at a given batch index, i.e something displayable.

        # Argument:
            - index: The index of the batch
        """
        if self.load_images_into_memory:
            return self.images[index]
        else:
            return np.array(Image.open(self.images_path[index]))

    def get_batch_data(self, index):
        """ Should return the data associated for the batch specified if any. Should return None else.

        # Argument:
            - index: The index of the batch
        """
        raise NotImplementedError(
            "Should return the data associated for the batch specified if any. Should return None else.")

    def shuffle_data(self):
        """ Should shuffle the batches of data."""
        shuffle(self.dataset_indices)

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

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  exclude_difficult=False,
                  ret=False):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, image IDs,
            and a list indicating which boxes are annotated with the label "difficult".
        '''

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                # Note: These are strings, not integers.
                image_ids = [line.strip() for line in f]
                self.image_ids += image_ids

            it = tqdm(image_ids, desc="Processing image set '{}'".format(
                os.path.basename(image_set_filename)), file=sys.stdout)

            # Loop over all images in this dataset.
            for image_id in it:

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                    soup = BeautifulSoup(f, 'xml')

                folder = soup.folder.text

                boxes = []  # We'll store all boxes for this image here.
                # We'll store whether a box is annotated as "difficult" here.
                eval_neutr = []
                # Get a list of all objects in this image.
                objects = soup.find_all('object')

                # Parse the data for each object.
                for obj in objects:
                    class_name = obj.find('name', recursive=False).text
                    class_id = self.classes.index(class_name)
                    pose = obj.find('pose', recursive=False).text
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
                    item_dict = {'folder': folder,
                                 'image_name': filename,
                                 'image_id': image_id,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'pose': pose,
                                 'truncated': truncated,
                                 'difficult': difficult,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    for item in self.labels_output_format:
                        box.append(item_dict[item])
                    boxes.append(box)
                    if difficult:
                        eval_neutr.append(True)
                    else:
                        eval_neutr.append(False)

                self.labels.append(boxes)
                self.eval_neutral.append(eval_neutr)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            it = tqdm(self.filenames,
                      desc='Loading images into memory', file=sys.stdout)
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

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

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False):
