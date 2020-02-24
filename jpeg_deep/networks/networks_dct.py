import os

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Dropout, Conv2DTranspose, Concatenate, GlobalAveragePooling2D
from keras import models
from keras.regularizers import l2

from jpeg_deep.layers import ResizeFeatures


def vgga_dct(classes=1000, input_shape=(28, 28)):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)
    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005), padding='same',
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vggd_dct(classes=1000, input_shape=(28, 28)):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vgga_dct_resize(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (None, None, 64)
    input_shape_cbcr = (None, None, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)
    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005), padding='same',
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = ResizeFeatures((7, 7))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vggd_dct_resize(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (None, None, 64)
    input_shape_cbcr = (None, None, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = ResizeFeatures((7, 7))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vgga_dct_conv(classes=1000, input_shape=None):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)
    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005), padding='same',
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3), activation='relu',
               kernel_regularizer=l2(0.0005),
               padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Conv2D(4096, (7, 7), activation='relu', name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vggd_dct_conv(classes=1000, input_shape=None):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    if input_shape is None:
        input_shape_y = (None, None, 64)
        input_shape_cbcr = (None, None, 128)
    else:
        input_shape_y = (*input_shape, 64)
        input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    norm_cbcr = BatchNormalization(
        name="b_norm_128", input_shape=input_shape_cbcr)(input_cbcr)

    # Block 1
    x = BatchNormalization(
        name="b_norm_64", input_shape=input_shape_y)(input_y)

    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block1_conv1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Conv2D(4096, (7, 7), activation='relu',
               name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu',
               name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax',
               name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def vggd_dct_deconv(classes=1000, input_shape=(28, 28)):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cb = (input_shape[0] // 2, input_shape[1] // 2, 64)
    input_shape_cr = (input_shape[0] // 2, input_shape[1] // 2, 64)

    input_y = Input(input_shape_y)
    input_cb = Input(shape=input_shape_cb)
    input_cr = Input(shape=input_shape_cr)

    cb = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.00005))(input_cb)
    cr = Conv2DTranspose(64, kernel_size=(2, 2), strides=2,
                         kernel_regularizer=l2(0.00005))(input_cr)

    x = Concatenate(axis=-1)([input_y, cb, cr])

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',
              kernel_regularizer=l2(0.0005), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cb, input_cr], outputs=x)


def vggd_dct_deconv_conv(classes=1000, input_shape=(28, 28)):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_shape_y = (*input_shape, 64)
    input_shape_cbcr = (input_shape[0] // 2, input_shape[1] // 2, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    cbcr = Conv2DTranspose(128, kernel_size=(
        2, 2), stride=2, kernel_regularizer=l2(0.0005))(input_cbcr)

    x = Concatenate([input_y, cbcr])

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv1_dct')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv1_dct')(concat)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               kernel_regularizer=l2(0.0005),
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Conv2D(4096, (7, 7), activation='relu', name='conv2d_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', name='conv2d_2')(x)
    x = Conv2D(classes, (1, 1), activation='softmax', name='conv2d_3')(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)
