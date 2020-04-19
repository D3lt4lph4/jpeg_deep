from typing import List, Tuple

from keras.regularizers import l2
from keras.layers import Conv2D, BatchNormalization, Activation, add


def identity_block(input_tensor: object, kernel_size: int, filters: List[int], stage: str, block, kernel_reg: int=0.00005):
    """ The identity block as described in the ResNet paper.

    # Arguments
        - input_tensor: The input tensor.
        - kernel_size: The size of the kernel to use for the convolutions
        - filters: A list with the size of the different filters, should contain three numbers
        - stage: The name of the stage, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
        - kernel_reg: The regularization factor to use

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


def conv_block(input_tensor: object,
               kernel_size: int,
               filters: List[int],
               stage: str,
               block: str,
               strides: Tuple[int, int]=(2, 2),
               kernel_reg=0.00005):
    """ The convolution block as described in the ResNet paper.

    # Arguments
        - input_tensor: The input tensor.
        - kernel_size: The size of the kernel to use for the convolutions
        - filters: A list with the size of the different filters, should contain three numbers
        - stage: The name of the stage, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
        - stride: The stride to use for the convolution
        - kernel_reg: The regularization factor to use

    # Returns
        Output tensor for the block.
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
