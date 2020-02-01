import os

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Dropout, Conv2DTranspose, Concatenate, GlobalAveragePooling2D
from keras.models import Model
from keras import models
from keras.regularizers import l2


def vgga(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (224, 224, 3)
    else:
        input_shape = (3, 224, 224)

    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1_a'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2_a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1_a'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2_a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu',
                    kernel_regularizer=l2(0.0005), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',
                    kernel_regularizer=l2(0.0005), name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model


def vggd(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (224, 224, 3)
    else:
        input_shape = (3, 224, 224)

    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block1_conv1', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block3_conv2'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block4_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block4_conv2'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block5_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block5_conv2'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0005),
                     name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu',
                    kernel_regularizer=l2(0.0005), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',
                    kernel_regularizer=l2(0.0005), name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model


def vgga_conv(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (224, 224, 3)
    else:
        input_shape = (3, 224, 224)

    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1_a'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2_a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1_a'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2_a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Conv2D(4096, (7, 7), activation='relu', name='conv2d_1'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu', name='conv2d_2'))
    model.add(Dropout(0.5))
    model.add(Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3'))
    model.add(GlobalAveragePooling2D())

    return model


def vggd_conv(classes=1000, input_shape=(None, None)):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (*input_shape, 3)
    else:
        input_shape = (3, *input_shape)

    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Conv2D(4096, (7, 7), activation='relu', name='conv2d_1'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu', name='conv2d_2'))
    model.add(Dropout(0.5))
    model.add(Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3'))
    model.add(GlobalAveragePooling2D())

    return model
