import numpy as np

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from template_tensorflow.generators import TemplateGenerator

class DCTGenerator(TemplateGenerator):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, image_shape=(28, 28), shuffle=True, num_classes=10):
        'Initialization'

        # Mandatory variables
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._number_of_data_samples = len(data[0])

        # Non mandatory stuff
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.x_train = data[0]
        self.y_train = to_categorical(data[1], self.num_classes)
        self.batches_per_epoch = int(np.floor(len(self.x_train) / self.batch_size))
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
    
    def _decode_and_random_crop(self, image_buffer, bbox, image_size):
        """Randomly crops image and then scales to target size."""
        with tf.name_scope('distorted_bounding_box_crop',
                            values=[image_buffer, bbox]):
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.image.extract_jpeg_shape(image_buffer),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.08, 1.0],
                max_attempts=10,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
        image = tf.image.convert_image_dtype(
            image, dtype=tf.float32)

        image = tf.image.resize_bicubic([image],
                                        [image_size, image_size])[0]

        return image

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % (self.batches_per_epoch)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def get_raw_input_label(self, index):
        return self.__getitem__(index)

    def shuffle_batches(self):
        """ Shuffle the batches of data."""
        np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.zeros((self.batch_size, *self.image_shape, 1))
        y = np.empty((self.batch_size, self.num_classes))

        for i in range(self.batch_size):
            X[i] = self.x_train[indexes[i]].reshape(*self.image_shape, 1)
            y[i] = self.y_train[indexes[i]]

        return X, y

    def get_batch_data(self, index):
        return None

class DummyGenerator(TemplateGenerator):
    'Generates dummy data to use to test the network speed'
    def __init__(self, num_batches, batch_size=32, image_shape=(28, 28), shuffle=True, num_classes=10):
        'Initialization'
        self.image_shape = image_shape
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.num_classes = num_classes
        self.batches_per_epoch = num_batches
        self.indexes = np.arange(num_batches)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def shuffle_batches(self):
        """ Shuffle the batches of data."""
        np.random.shuffle(self.indexes)
    
    def get_batch_data(self, index):
        return None

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
        index = index % (self.batches_per_epoch)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.image_shape, 1))
        y = np.empty((self.batch_size, self.num_classes))

        return X, y
    def get_raw_input_label(self, index):
        return self.__getitem__(index)