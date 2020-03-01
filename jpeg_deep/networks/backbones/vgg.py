from typing import Tuple

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from keras.regularizers import l2


def feature_map_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the RGB network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    img_h, img_w = image_shape
    input_layer = Input(shape=(img_h, img_w, 3))

    block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block1_conv1')(input_layer)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block1_pool')(block1_conv2)

    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block2_pool')(block2_conv2)

    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block3_pool')(block3_conv3)

    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block4_pool')(block4_conv3)
    return input_layer, block4_pool, block4_conv3


def feature_map_dct(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    if image_shape[0] is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        img_h, img_w = image_shape
        input_shape_y = (img_h, img_w, 64)
        input_shape_cbcr = (img_h / 2, img_w / 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_cbcr", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    norm_y = BatchNormalization(
        name="b_norm_y", input_shape=input_shape_y)(input_y)

    block1_conv1 = Conv2D(256, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block1_conv1_dct_256')(norm_y)

    # Block 4
    block4_conv1 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv1_dct')(block1_conv1)
    block4_conv2 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool')(block4_conv3)

    concat = Concatenate(axis=-1, name="concat_dct")([block4_pool, norm_cbcr])

    return [input_y, input_cbcr], concat, block4_conv3
