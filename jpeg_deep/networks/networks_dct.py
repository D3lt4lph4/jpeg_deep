import os

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Dropout, Conv2DTranspose, Concatenate
from keras import models


def vgga_dct(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)
    # Block 1
    x = BatchNormalization(name="b_norm_64", input_shape=input_shape_y)(input_y)
    
    x = Conv2D(256, (3, 3),
            activation='relu',
            padding='same',
            name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


    concat = Concatenate(axis=-1)([x, norm_cbcr])
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vggd_dct(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(name="b_norm_64", input_shape=input_shape_y)(input_y)
    
    x = Conv2D(256, (3, 3),
            activation='relu',
            padding='same',
            name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vgga_dct_8x8(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into.
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (224, 224, 3)
    else:
        input_shape = (3, 224, 224)

    model = Sequential()

    # Block 1
    model.add(BatchNormalization(name="b_norm_input"))
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1_dct_8x8',
               input_shape=input_shape))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1_dct'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model


def vggd_dct_8x8(classes=1000):
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
    model.add(BatchNormalization(name="b_norm_input"))
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1_dct_8x8',
               input_shape=input_shape))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1_dct'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2'))
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model
