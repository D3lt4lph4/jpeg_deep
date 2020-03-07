from typing import Tuple

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate, BatchNormalization, ZeroPadding2D, Conv2DTranspose
from keras.regularizers import l2

from jpeg_deep.layers.ssd_layers import AnchorBoxes, L2Normalization, DecodeDetections

from ..resnet_blocks import conv_block, identity_block


def feature_map_resnet_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'he_normal', l2_reg=0.0005):
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

    # 1
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg),
               name='conv1')(x)
    x = BatchNormalization(
        axis=3, momentum=0.9, epsilon=1e-5, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a',
                   strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [64, 64, 256], stage=2,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [64, 64, 256], stage=2,
                       block='c', kernel_reg=l2_reg)

    # 3
    x = conv_block(x, 3, [128, 128, 512], stage=3,
                   block='a', kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='c', kernel_reg=l2_reg)
    block4_conv3 = identity_block(
        x, 3, [128, 128, 512], stage=3, block='d', kernel_reg=l2_reg)

    # 4
    x = conv_block(block4_conv3, 3, [256, 256, 1024],
                   stage=4, block='a', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='c', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='d', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='e', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='f', kernel_reg=l2_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5,
                   block='a', kernel_reg=l2_reg, strides=(1, 1))
    x = identity_block(x, 3, [512, 512, 2048], stage=5,
                       block='b', kernel_reg=l2_reg)
    last = identity_block(x, 3, [512, 512, 2048],
                          stage=5, block='c', kernel_reg=l2_reg)

    return input_layer, last, block4_conv3


def feature_map_lcrfa(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(shape=input_shape_y)
    input_cbcr = Input(shape=input_shape_cbcr)

    bn_y = BatchNormalization(
        axis=3, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    y = conv_block(bn_y, 1, [256, 256, 1024],
                   stage=1, block="a_y", strides=(1, 1), kernel_reg=l2_reg)
    y = identity_block(y, 2, [256, 256, 1024], stage=1,
                       block="b_y", kernel_reg=l2_reg)
    y = identity_block(y, 3, [256, 256, 1024], stage=1,
                       block="c_y", kernel_reg=l2_reg)

    y = conv_block(y, 3, [128, 128, 512], stage=2,
                   block='a_y', strides=(1, 1), kernel_reg=l2_reg)
    y = identity_block(y, 3, [128, 128, 512], stage=2,
                       block='b_y', kernel_reg=l2_reg)
    y = identity_block(y, 3, [128, 128, 512], stage=2,
                       block='c_y', kernel_reg=l2_reg)
    block4_conv3 = identity_block(
        y, 3, [128, 128, 512], stage=2, block='d_y', kernel_reg=l2_reg)

    y = conv_block(block4_conv3, 3, [128, 128, 512],
                   stage=3, block='a_y', kernel_reg=l2_reg)

    bn_cbcr = BatchNormalization(
        axis=3, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(input_cbcr)
    cbcr = conv_block(bn_cbcr, 1, [128, 128, 512],
                      stage=1, block='a_cbcr', strides=(1, 1), kernel_reg=l2_reg)

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='c', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='d', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='e', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='f', kernel_reg=l2_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5,
                   block='a', strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,
                       block='b', kernel_reg=l2_reg)
    last = identity_block(x, 3, [512, 512, 2048],
                          stage=5, block='c', kernel_reg=l2_reg)

    return [input_y, input_cbcr], last, block4_conv3


def feature_map_lcrfat(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(shape=input_shape_y)
    input_cbcr = Input(shape=input_shape_cbcr)

    bn_y = BatchNormalization(
        axis=3, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    y = conv_block(bn_y, 1, [256, 256, 384],
                   stage=1, block="a_y", strides=(1, 1), kernel_reg=l2_reg)
    y = identity_block(y, 2, [256, 256, 384], stage=1,
                       block="b_y", kernel_reg=l2_reg)
    y = identity_block(y, 3, [256, 256, 384], stage=1,
                       block="c_y", kernel_reg=l2_reg)

    y = conv_block(y, 3, [128, 128, 384], stage=2,
                   block='a_y', strides=(1, 1), kernel_reg=l2_reg)
    y = identity_block(y, 3, [128, 128, 384], stage=2,
                       block='b_y', kernel_reg=l2_reg)
    y = identity_block(y, 3, [128, 128, 384], stage=2,
                       block='c_y', kernel_reg=l2_reg)
    block4_conv3 = identity_block(
        y, 3, [128, 128, 384], stage=2, block='d_y', kernel_reg=l2_reg)

    y = conv_block(block4_conv3, 3, [128, 128, 768],
                   stage=3, block='a_y', kernel_reg=l2_reg)

    bn_cbcr = BatchNormalization(
        axis=3, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(input_cbcr)
    cbcr = conv_block(bn_cbcr, 1, [128, 128, 256],
                      stage=1, block='a_cbcr', strides=(1, 1), kernel_reg=l2_reg)

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='c', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='d', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='e', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='f', kernel_reg=l2_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5,
                   block='a', strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,
                       block='b', kernel_reg=l2_reg)
    last = identity_block(x, 3, [512, 512, 2048],
                          stage=5, block='c', kernel_reg=l2_reg)

    return [input_y, input_cbcr], last, block4_conv3


def feature_map_deconvolution_rfa(image_shape: Tuple[int, int],  kernel_initializer: str = 'he_normal', l2_reg=0.0005):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    input_shape_y = (38, 38, 64)
    input_shape_cb = (19, 19, 64)
    input_shape_cr = (19, 19, 64)

    input_y = Input(shape=input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(l2_reg))(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(l2_reg))(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])
    x = BatchNormalization(axis=3, momentum=0.9,
                           epsilon=1e-5, name='bn_cbcr')(x)

    x = conv_block(x, 1, [256, 256, 1024],
                   stage=1, block="a", strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 2, [256, 256, 1024], stage=2,
                       block="b", kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=2,
                       block="c", kernel_reg=l2_reg)

    x = conv_block(x, 3, [128, 128, 512], stage=3,
                   block='a', strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='c', kernel_reg=l2_reg)
    block4_conv3 = identity_block(
        y, 3, [128, 128, 512], stage=3, block='d', kernel_reg=l2_reg)

    x = conv_block(block4_conv3, 3, [256, 256, 1024],
                   stage=4, block='a', kernel_reg=l2_reg)

    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='c', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='d', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='e', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       block='f', kernel_reg=l2_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5,
                   block='a', strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,
                       block='b', kernel_reg=l2_reg)
    last = identity_block(x, 3, [512, 512, 2048],
                          stage=5, block='c', kernel_reg=l2_reg)

    return [input_y, input_cbcr], last, block4_conv3
