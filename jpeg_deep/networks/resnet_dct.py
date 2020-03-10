"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
import os
import warnings

from keras.regularizers import l2
from keras.layers import Input, BatchNormalization, Concatenate, GlobalAveragePooling2D, Dense, Conv2DTranspose
from keras.models import Model

from .resnet_blocks import identity_block, conv_block


def late_concat_rfa(input_shape=(28, 28), classes: int = 1000):
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(shape=input_shape_y)
    input_cbcr = Input(shape=input_shape_cbcr)

    bn_axis = 3

    bn_y = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    y = conv_block(bn_y, 1, [256, 256, 1024],
                   stage=1, block="a_y", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 1024], stage=1, block="b_y")
    y = identity_block(y, 3, [256, 256, 1024], stage=1, block="c_y")

    y = conv_block(y, 3, [128, 128, 512], stage=2, block='a_y', strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='b_y')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='c_y')
    y = identity_block(y, 3, [128, 128, 512], stage=2, block='d_y')

    y = conv_block(y, 3, [128, 128, 512], stage=3, block='a_y')

    bn_cbcr = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(input_cbcr)
    cbcr = conv_block(bn_cbcr, 1, [128, 128, 512],
                      stage=1, block='a_cbcr', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax',
              kernel_regularizer=l2(0.00005), name='fc1000')(x)

    model = Model([input_y, input_cbcr], x,
                  name='resnet50_late_concat_rfa')

    return model


def late_concat_rfa_thinner(input_shape=(28, 28), classes=1000):
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(shape=input_shape_y)
    input_cbcr = Input(shape=input_shape_cbcr)

    bn_axis = 3

    bn_y = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    y = conv_block(bn_y, 1, [256, 256, 384],
                   stage=1, block="a_y", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 384], stage=1, block="b_y")
    y = identity_block(y, 3, [256, 256, 384], stage=1, block="c_y")

    y = conv_block(y, 3, [128, 128, 384], stage=2, block='a_y', strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='b_y')
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='c_y')
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='d_y')

    y = conv_block(y, 3, [128, 128, 768], stage=3, block='a_y')

    bn_cbcr = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_cbcr')(input_cbcr)
    cbcr = conv_block(bn_cbcr, 1, [128, 128, 256],
                      stage=1, block='a_cbcr', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax',
              kernel_regularizer=l2(0.00005), name='fc1000')(x)

    # Create model.
    model = Model([input_y, input_cbcr], x,
                  name='resnet50_late_concat_rfa_thinner')

    return model


def deconvolution_rfa(input_shape=(28, 28), classes: int = 1000):
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cb = (None, None, 64)
        input_shape_cr = (None, None, 64)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cb = (input_shape[0] // 2, input_shape[1] // 2, 64)
        input_shape_cr = (input_shape[0] // 2, input_shape[1] // 2, 64)

    input_y = Input(shape=input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.00005))(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.00005))(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])
    x = BatchNormalization(axis=3, momentum=0.9,
                           epsilon=1e-5, name='bn_cbcr')(x)

    x = conv_block(x, 1, [256, 256, 1024], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 2, [256, 256, 1024], stage=2, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax',
              kernel_regularizer=l2(0.00005), name='fc1000')(x)

    model = Model([input_y, input_cb, input_cr], x,
                  name='resnet50_deconv_rfa')

    return model


def late_concat_rfa_y(input_shape=(28, 28), classes: int = 1000):
    if input_shape is None:
        input_shape_y = (None, None, 64)
    else:
        input_shape_y = (*input_shape, 64)

    input_y = Input(shape=input_shape_y)

    bn_axis = 3

    bn_y = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    x = conv_block(bn_y, 1, [256, 256, 1024],
                   stage=1, block="a_y", strides=(1, 1))
    x = identity_block(x, 2, [256, 256, 1024], stage=1, block="b_y")
    x = identity_block(x, 3, [256, 256, 1024], stage=1, block="c_y")

    x = conv_block(x, 3, [128, 128, 512], stage=2, block='a_y', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='b_y')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='c_y')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='d_y')

    x = conv_block(x, 3, [128, 128, 1024], stage=3, block='a_y')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax',
              kernel_regularizer=l2(0.00005), name='fc1000')(x)

    model = Model(input_y, x,
                  name='resnet50_late_concat_rfa_y')

    return model


def late_concat_rfa_y_thinner(input_shape=(28, 28), classes=1000):
    if input_shape is None:
        input_shape_y = (None, None, 64)
    else:
        input_shape_y = (*input_shape, 64)

    input_y = Input(shape=input_shape_y)

    bn_axis = 3

    bn_y = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name='bn_y')(input_y)

    x = conv_block(bn_y, 1, [256, 256, 384],
                   stage=1, block="a_y", strides=(1, 1))
    x = identity_block(x, 2, [256, 256, 384], stage=1, block="b_y")
    x = identity_block(x, 3, [256, 256, 384], stage=1, block="c_y")

    x = conv_block(x, 3, [128, 128, 384], stage=2, block='a_y', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 384], stage=2, block='b_y')
    x = identity_block(x, 3, [128, 128, 384], stage=2, block='c_y')
    x = identity_block(x, 3, [128, 128, 384], stage=2, block='d_y')

    x = conv_block(x, 3, [128, 128, 1024], stage=3, block='a_y')

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax',
              kernel_regularizer=l2(0.00005), name='fc1000')(x)

    # Create model.
    model = Model(input_y, x,
                  name='resnet50_late_concat_rfa_y_thinner')

    return model
