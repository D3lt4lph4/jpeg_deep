import numpy as np
import keras
import os
import json
import random

from io import BytesIO

import cv2

from PIL import Image
from jpeg2dct.numpy import load, loads

from template_keras.generators import TemplateGenerator

class GeneratorPreResized(TemplateGenerator):
    'Generates data in the DCT space for Keras.'

    def __init__(self, data_directory, index_file, batch_size=32, shuffle=True, load_in_memory=True):

        self.batch_size = batch_size
        self.data_directory = data_directory
        self.shuffle = shuffle
        self.load_in_memory = load_in_memory

        # Process the index dictionary to get the matching name/class_id
        self.association = {}
        with open(index_file) as index:
            data = json.load(index)
            for id, value in data.items():
                self.association[value[0]] = id

        # First we process the data directory to get all the classes
        self.classes = []
        self.images_path = []
        self.data = []
        for directory in os.listdir(data_directory):
            class_directory = os.path.join(data_directory, directory)
            if os.path.isdir(class_directory):
                self.classes.append(directory)
                for image in os.listdir(class_directory):
                    image_path = os.path.join(class_directory, image)
                    self.images_path.append(image_path)
                    if load_in_memory:
                        # All the images would not fit in memory so we just load the compressed data
                        with open(image_path, "rb") as fin:
                            self.data.append(fin.read())

        self.number_of_classes = len(self.classes)
        
        self.batches_per_epoch = len(self.images_path) // self.batch_size
        self.indexes = np.arange(len(self.images_path))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % self.batches_per_epoch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'

        # Two inputs for the data of one image.
        X_y = np.empty((self.batch_size, 28, 28, 64))
        X_cbcr = np.empty((self.batch_size, 14, 14, 128))
        y = np.zeros((self.batch_size, self.number_of_classes))

        # iterate over the indexes to get the correct values
        for i, k in enumerate(indexes):

            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]

            # if the data is already loaded in memory, we read from it
            if self.load_in_memory:
                dct_y, dct_cb, dct_cr = loads(self.data[k])
            else:
                dct_y, dct_cb, dct_cr = load(self.images_path[k])
            
            # We carry the "data-augmentation"
            # For this generator, all the images have one size at 224, find it and then random select the 224 pixels from the other side
            if dct_y.shape[0] == 28:
                biggest = dct_y.shape[1]
            else:
                biggest = dct_y.shape[0]
            
            offset = random.randint(0, biggest - 28)

            # Load the data in the matrices
            if dct_y.shape[0] == 28:
                X_y[i] = dct_y[:,offset:28 + offset, :]
                X_cbcr[i] = np.concatenate((dct_cb[:,offset:14 + offset, :], dct_cr[:,offset:14 + offset, :]), -1)
            else:
                X_y[i] = dct_y[offset:28 + offset, :, :]
                X_cbcr[i] = np.concatenate((dct_cb[offset:28 + offset, :, :], dct_cr[offset:28 + offset, :, :]), -1)

            y[i, int(self.association[index_class])] = 1

        return [X_y, X_cbcr], y

class DCTGeneratorJPEG2DCT_111(TemplateGenerator):
    'Generates data in the DCT space for Keras.'

    def __init__(self, data_directory, index_file, batch_size=32, shuffle=True, scale=True, target_length=224):

        self._batch_size = batch_size
        self.data_directory = data_directory
        self._shuffle = shuffle
        self.scale = scale
        self.target_length = target_length

        # Process the index dictionary to get the matching name/class_id
        self.association = {}
        with open(index_file) as index:
            data = json.load(index)
            for id, value in data.items():
                self.association[value[0]] = id

        # First we process the data directory to get all the classes
        self.classes = []
        self.images_path = []
        self.data = []
        for directory in os.listdir(data_directory):
            class_directory = os.path.join(data_directory, directory)
            if os.path.isdir(class_directory):
                self.classes.append(directory)
                for image in os.listdir(class_directory):
                    image_path = os.path.join(class_directory, image)
                    self.images_path.append(image_path)
                    
        self.number_of_classes = len(self.classes)

        self._number_of_data_samples = len(self.images_path)
        
        self.batches_per_epoch = len(self.images_path) // self._batch_size
        self.indexes = np.arange(len(self.images_path))
        self.on_epoch_end()
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def number_of_data_samples(self):
        return self._number_of_data_samples
    
    @number_of_data_samples.setter
    def number_of_data_samples(self, value):
        self._number_of_data_samples = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % self.batches_per_epoch
        indexes = self.indexes[index*self.batch_size:(index+1)*self._batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self._shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'

        # Two inputs for the data of one image.
        X = np.empty((self._batch_size, 28, 28, 192), dtype=np.int32)
        y = np.zeros((self._batch_size, self.number_of_classes))

        # iterate over the indexes to get the correct values
        for i, k in enumerate(indexes):

            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]

            # Load the image in RGB,
            with Image.open(self.images_path[k]) as im:
                if self.scale:
                    min_side = min(im.size)
                    scaling_ratio = self.target_length / min_side

                    width, height = im.size
                    im.resize((int(round(width * scaling_ratio)), int(round(height * scaling_ratio))))
                else:
                    im.resize((int(self.target_length), int(self.target_length)))

                if self.scale:
                    offset = random.randint(0, max(im.size) - self.target_length)
                    
                    if im.size[0] > im.size[1]:
                        im = im.crop((offset, 0, self.target_length + offset, self.target_length))
                    else:
                        im = im.crop((0, offset, self.target_length, self.target_length + offset))
                fake_file = BytesIO()
                print("hey")
                im.save(fake_file, format="jpeg", subsampling=0)

            dct_y, dct_cb, dct_cr = loads(fake_file.getvalue())

            print(dct_y.shape)
            print(dct_cb.shape)
            print(dct_cr.shape)

            X[i] = np.concatenate([dct_y, dct_cb, dct_cr])

            y[i, int(self.association[index_class])] = 1

        return X, y

class DCTGeneratorImageNet(TemplateGenerator):
    'Generates data in the DCT space for Keras.'

    def __init__(self, data_directory, index_file, batch_size=32, image_shape=(224, 224, 3), shuffle=True, target_length=224):
        'Initialization'
        self.image_shape = image_shape
        self._batch_size = batch_size
        self.data_directory = data_directory
        self.decoder = jpegdecoder.decoder.JPEGDecoder()
        self.target_length = target_length

        # Process the index dictionary to get the matching name/class_id
        self.association = {}
        with open(index_file) as index:
            data = json.load(index)
            for id, value in data.items():
                self.association[value[0]] = id

        # First we process the data directory to get all the classes
        self.classes = []
        self.images_path = []
        for directory in os.listdir(data_directory):
            class_directory = os.path.join(data_directory, directory)
            if os.path.isdir(class_directory):
                self.classes.append(directory)
                for image in os.listdir(class_directory):
                    self.images_path.append(
                        os.path.join(class_directory, image))

        self.number_of_classes = len(self.classes)
        self._shuffle = shuffle
        self.batches_per_epoch = len(self.images_path) // self.batch_size
        self.indexes = np.arange(len(self.images_path))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % self.batches_per_epoch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_images_path = []
        # Find list of IDs
        for k in indexes:
            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]
            batch_images_path.append(
                (self.images_path[k], self.association[index_class]))

        # Generate data
        X, y = self.__data_generation(batch_images_path)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_images_path):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization

        X = np.empty((self.batch_size, *self.image_shape))
        y = np.zeros((self.batch_size, self.number_of_classes))

        # We load into memory the corresponding images
        for i, image_path in enumerate(batch_images_path):
            # Store sample
            img = self.decoder.decode_file(image_path[0], 2)
            rows, cols = img.get_component_shape(0)[0:2]
            if img.get_number_of_component() == 1:
                X[i, :, :, 0] = np.reshape(img.get_data(0), (rows, cols))[
                    :self.image_shape[0], :self.image_shape[1]]
                X[i, :, :, 1] = X[i, :, :, 0]
                X[i, :, :, 2] = X[i, :, :, 0]
            else:
                X[i, :, :, 0] = np.reshape(img.get_data(0), (rows, cols))[
                    :self.image_shape[0], :self.image_shape[1]]
                X[i, :, :, 1] = np.reshape(img.get_data(1), (rows, cols))[
                    :self.image_shape[0], :self.image_shape[1]]
                X[i, :, :, 2] = np.reshape(img.get_data(2), (rows, cols))[
                    :self.image_shape[0], :self.image_shape[1]]

            y[i, int(image_path[1])] = 1

        return X, y

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def shuffle(self):
        return self._shuffle

class DummyGenerator(TemplateGenerator):
    'Generates data in the DCT space for Keras.'

    def __init__(self, num_batches, batch_size=32, number_of_classes=1000, image_shape=(224, 224, 3), shuffle=True):
        'Initialization'
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.batches_per_epoch = num_batches
        self.number_of_classes = number_of_classes
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization

        X = np.empty((self.batch_size, *self.image_shape))
        y = np.empty((self.batch_size, self.number_of_classes))

        return X, y
