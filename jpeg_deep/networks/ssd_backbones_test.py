from typing import Tuple

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate, BatchNormalization, ZeroPadding2D
from keras.regularizers import l2

from jpeg_deep.layers.ssd_layers import AnchorBoxes, L2Normalization, DecodeDetections


def feature_map_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'he_normal', l2_reg=0.00025):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the RGB network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, conv4_pool, conv4_3. These layers are used to intantiate the network.
    """
    img_h, img_w = image_shape
    input_layer = Input(shape=(img_h, img_w, 3))

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv1_1')(input_layer)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv1_2')(conv1_1)
    conv1_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='conv1_pool')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv2_1')(conv1_pool)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv2_2')(conv2_1)
    conv2_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='conv2_pool')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv3_1')(conv2_pool)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv3_3')(conv3_2)
    conv3_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='conv3_pool')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv4_1')(conv3_pool)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          kernel_initializer=kernel_initializer, name='conv4_3')(conv4_2)
    conv4_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='conv4_pool')(conv4_3)
    return input_layer, conv4_pool, conv4_3


def feature_map_dct(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.00025):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, conv4_pool, conv4_3. These layers are used to intantiate the network.
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

    conv1_1 = Conv2D(256, (3, 3), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='conv1_1_dct_256')(norm_y)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='conv4_1_dct')(conv1_1)
    conv4_2 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          activation='relu',
                          padding='same',
                          name='conv4_3')(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(
        2, 2), name='conv4_pool')(conv4_3)

    concat = Concatenate(axis=-1, name="concat_dct")([conv4_pool, norm_cbcr])

    return [input_y, input_cbcr], concat, conv4_3


def feature_map_resnet_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'he_normal', l2_reg=0.00025):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the RGB network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, conv4_pool, conv4_3. These layers are used to intantiate the network.
    """
    img_h, img_w = image_shape
    input_layer = Input(shape=(img_h, img_w, 3))

    # 1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2
    x = conv_conv(x, 3, [64, 64, 256], stage=2, conv='a', strides=(1, 1))
    x = identity_conv(x, 3, [64, 64, 256], stage=2, conv='b')
    conv4_3 = identity_conv(x, 3, [64, 64, 256], stage=2, conv='c')

    # 3
    x = conv_conv(conv4_3, 3, [128, 128, 512], stage=3, conv='a')
    x = identity_conv(x, 3, [128, 128, 512], stage=3, conv='b')
    x = identity_conv(x, 3, [128, 128, 512], stage=3, conv='c')
    conv4_pool = identity_conv(x, 3, [128, 128, 512], stage=3, conv='d')

    return input_layer, conv4_pool, conv4_3


def feature_map_resnet_dct(image_shape: Tuple[int, int],  kernel_initializer: str = 'glorot_uniform'):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, conv4_pool, conv4_3. These layers are used to intantiate the network.
    """
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    Y = layers.Input(shape=input_shape_y)
    CbCr = layers.Input(shape=input_shape_cbcr)

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_Y = layers.BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_y')(Y)

    y = conv_conv(bn_Y, 1, [256, 256, 1024],
                   stage=1, conv="a", strides=(1, 1))
    y = identity_conv(y, 2, [256, 256, 1024], stage=1, conv="b")
    y = identity_conv(y, 3, [256, 256, 1024], stage=1, conv="c")

    y = conv_conv(y, 3, [128, 128, 512], stage=2, conv='a', strides=(1, 1))
    y = identity_conv(y, 3, [128, 128, 512], stage=2, conv='b')
    y = identity_conv(y, 3, [128, 128, 512], stage=2, conv='c')
    conv4_3 = identity_conv(y, 3, [128, 128, 512], stage=2, conv='d')

    y = conv_conv(conv4_3, 3, [256, 256, 512], stage=2, conv='a4')

    bn_CbCr = layers.BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(CbCr)
    cbcr = conv_conv(bn_CbCr, 1, [256, 256, 1024],
                      stage=2, conv='a5', strides=(1, 1))

    concat = Concatenate(axis=-1)([y, cbcr])

    return [Y, CbCr], concat, conv4_3
