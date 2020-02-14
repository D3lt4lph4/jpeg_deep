"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
import os
import warnings

from keras.regularizers import l2
from keras import layers
from keras import backend
from keras import models

from .resnet_blocks import identity_block, conv_block


def late_concat_rfa():
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

    y = conv_block(bn_Y, 1, [256, 256, 1024],
                   stage=1, block="a", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 1024], stage=1, block="b")
    y = identity_block(y, 3, [256, 256, 1024], stage=1, block="c")

    y = conv_block(y, 3, [128, 128, 512], stage=2, block='a', strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='b')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='c')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='d')

    y = conv_block(y, 3, [256, 256, 512], stage=2, block='a4')

    bn_CbCr = layers.BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(CbCr)
    cbcr = conv_block(bn_CbCr, 1, [256, 256, 1024],
                      stage=2, block='a5', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax',
                     kernel_regularizer=l2(0.00005), name='fc1000')(x)

    inputs = [img_input, img_input]

    # Create model.
    model = models.Model(inputs, x, name='resnet50_late_concat_rfa')


def late_concat_rfa_thinner():
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

    y = conv_block(bn_Y, 1, [256, 256, 384],
                   stage=1, block="a", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 1024], stage=1, block="b")
    y = identity_block(y, 3, [256, 256, 1024], stage=1, block="c")

    y = conv_block(y, 3, [128, 128, 384], stage=2, block='a', strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='b')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='c')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='d')

    y = conv_block(y, 3, [256, 256, 768], stage=2, block='a4')

    bn_CbCr = layers.BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(CbCr)
    cbcr = conv_block(bn_CbCr, 1, [256, 256, 1024],
                      stage=2, block='a5', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax',
                     kernel_regularizer=l2(0.00005), name='fc1000')(x)

    inputs = [img_input, img_input]

    # Create model.
    model = models.Model(inputs, x, name='resnet50_late_concat_rfa_thinner')
