import os
import warnings

from keras.regularizers import l2
from keras.layers import Conv2D, BatchNormalization, Activation, add


def identity_block(input_tensor, kernel_size, filters, stage, block, kernel_reg=0.00005):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)
               kernel_reg=0.00005):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               kernel_regularizer=l2(kernel_reg),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(kernel_reg),
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, momentum=0.9, epsilon=1e-5, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x
