from typing import Tuple

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Conv2DTranspose, Lambda
from keras.regularizers import l2

from jpeg_deep.layers import ResizeFeatures


def feature_map_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'he_normal', l2_reg=0.0005, rescale_position: int = 0):
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


def identity_layer(tensor):
    return tensor


def feature_map_dct(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005, rescale_position: int = 0):
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

    if rescale_position == 10:
        input_y_l = Lambda(identity_layer, output_shape=(
            None, None, 64), name='identity_layer_l')(input_y)
        input_cbcr_l = Lambda(identity_layer, output_shape=(
            None, None, 128), name='identity_layer')(input_cbcr)

        input_y_l = ResizeFeatures((38, 38))(input_y_l)
        input_cbcr_l = ResizeFeatures((19, 19))(input_cbcr_l)

        norm_cbcr = BatchNormalization(
            name="b_norm_cbcr", input_shape=input_shape_cbcr)(input_cbcr_l)
        # Block 1
        norm_y = BatchNormalization(
            name="b_norm_y", input_shape=input_shape_y)(input_y_l)
    else:
        norm_cbcr = BatchNormalization(
            name="b_norm_cbcr", input_shape=input_shape_cbcr)(input_cbcr)
        # Block 1
        norm_y = BatchNormalization(
            name="b_norm_y", input_shape=input_shape_y)(input_y)

    if 0 < rescale_position <= 5:
        norm_cbcr = ResizeFeatures((19, 19))(norm_cbcr)

    if rescale_position == 1:
        norm_y = ResizeFeatures((38, 38))(norm_y)

    block1_conv1 = Conv2D(256, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block1_conv1_dct_256')(norm_y)

    if rescale_position == 2:
        block1_conv1 = ResizeFeatures((38, 38))(block1_conv1)

    # Block 4
    block4_conv1 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv1_dct')(block1_conv1)
    if rescale_position == 3:
        block4_conv1 = ResizeFeatures((38, 38))(block4_conv1)

    block4_conv2 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(block4_conv1)

    if rescale_position == 4:
        block4_conv2 = ResizeFeatures((38, 38))(block4_conv2)

    block4_conv3 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(block4_conv2)
    if rescale_position == 6:
        block4_conv3_out = ResizeFeatures((38, 38), method="nn")(block4_conv3)
        block4_pool = MaxPooling2D((2, 2), strides=(
            2, 2), name='block4_pool')(block4_conv3)
    else:
        block4_conv3_out = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg),
                                  activation='relu',
                                  padding='same',
                                  name='block4_conv3')(block4_conv2)
        if rescale_position == 5:
            block4_conv3_out = ResizeFeatures((38, 38))(block4_conv3_out)
        block4_pool = MaxPooling2D((2, 2), strides=(
            2, 2), name='block4_pool')(block4_conv3_out)

    concat = Concatenate(axis=-1, name="concat_dct")([block4_pool, norm_cbcr])

    return [input_y, input_cbcr], concat, block4_conv3_out


def feature_map_dct_deconv(input_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cb = (input_shape[0] // 2, input_shape[1] // 2, 64)
    input_shape_cr = (input_shape[0] // 2, input_shape[1] // 2, 64)

    input_y = Input(input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(l2_reg))(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(l2_reg))(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])

    x = BatchNormalization(
        name="b_norm", input_shape=input_shape_y)(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(l2_reg),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(l2_reg),
               name='block4_conv2')(x)
    block4_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          kernel_regularizer=l2(l2_reg),
                          name='block4_conv3')(x)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    return [input_y, input_cb, input_cr], block4_pool, block4_conv3
