import os

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras import models

def vggA_dct(classes=1000, training=True):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_y = Input((28, 28, 64))
    input_cbcr = Input((14, 14, 128))

    # First block, y components
    batchnom_y = BatchNormalization(name="batchnorm_y")(input_y)

    conv_y_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_y)

    # First block, cbcr components
    batchnom_cbcr = BatchNormalization(name="batchnorm_y")(input_cbcr)

    conv_cbcr_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_cbcr)

    # Plugging back to the fourth block
    conv4_1 = Conv2D(512, (3, 3), activation='relu',
                      padding='same',
                      name='block4_conv1')(conv_y_1)
    conv4_2 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_2)

    # Block 5
    concat5 = Concatenate([pool4, conv_cbcr_1], -1, name="concat5")

    conv5_1 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_2')(conv5_1)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, training=training)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, training=training)(x)
    x = Dense(classes, activation='softmax', name='predictions_vgg16a')(x)

    return Model([input_y, input_cbcr], x)

def vggA_dct_batchnormed(classes=1000, training=True):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_y = Input((28, 28, 64))
    input_cbcr = Input((14, 14, 128))

    # First block, y components
    batchnom_y = BatchNormalization(name="batchnorm_y")(input_y)

    conv_y_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_y)

    # First block, cbcr components
    batchnom_cbcr = BatchNormalization(name="batchnorm_cbcr")(input_cbcr)

    conv_cbcr_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_cbcr)

    # Plugging back to the fourth block
    conv4_1 = Conv2D(512, (3, 3), activation='relu',
                      padding='same',
                      name='block4_conv1')(conv_y_1)
    conv4_1 = BatchNormalization(name="batchnorm_conv4_1")(conv4_1)

    conv4_2 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(conv4_1)
    conv4_2 = BatchNormalization(name="batchnorm_conv4_2")(conv4_2)

    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_2)

    # Block 5
    concat5 = Concatenate([pool4, conv_cbcr_1], -1, name="concat5")

    conv5_1 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_1')(pool4)
    conv5_1 = BatchNormalization(name="batchnorm_conv5_1")(conv5_1)

    conv5_2 = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_2')(conv5_1)
    conv5_2 = BatchNormalization(name="batchnorm_conv5_2")(conv5_2)

    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, training=training)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, training=training)(x)
    x = Dense(classes, activation='softmax', name='predictions_vgga')(x)

    return Model([input_y, input_cbcr], x)

def vggD_dct(classes=1000, training=True):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_y = Input((28, 28, 64), name="input_y")
    input_cbcr = Input((14, 14, 128), name="input_cbcr")

    # First block, y components
    batchnom_y = BatchNormalization(name="batchnorm_y")(input_y)

    conv_y_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_y)

    # First block, cbcr components
    batchnom_cbcr = BatchNormalization(name="batchnorm_cbcr")(input_cbcr)

    conv_cbcr_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_cbcr)

    # Plugging back to the fourth block
    x = Conv2D(512, (3, 3), activation='relu',
                      padding='same',
                      name='conv4_1')(conv_y_1)

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv4_2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='')(x)

    # Block 5
    concat5 = Concatenate([x, conv_cbcr_1], -1, name="concat5")

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_1')(concat5)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, training=True)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, training=True)(x)
    x = Dense(classes, activation='softmax', name='predictions_vggd')(x)

    return Model([input_y, input_cbcr], x)

def vggD_dct_batchnormed(classes=1000, training=True):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    input_y = Input((28, 28, 64), name="input_y")
    input_cbcr = Input((14, 14, 128), name="input_cbcr")

    # First block, y components
    batchnom_y = BatchNormalization(name="batchnorm_y")(input_y)

    conv_y_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_y)

    # First block, cbcr components
    batchnom_cbcr = BatchNormalization(name="batchnorm_cbcr")(input_cbcr)

    conv_cbcr_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_y_1')(batchnom_cbcr)

    # Plugging back to the fourth block
    x = Conv2D(512, (3, 3), activation='relu',
                      padding='same',
                      name='conv4_1')(conv_y_1)
    x = BatchNormalization(name="batchnorm_conv4_1")(x)

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv4_2')(x)
    x = BatchNormalization(name="batchnorm_conv4_2")(x)
    
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv4_3')(x)
    x = BatchNormalization(name="batchnorm_conv4_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='')(x)

    # Block 5
    concat5 = Concatenate([x, conv_cbcr_1], -1, name="concat5")

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_1')(concat5)
    x = BatchNormalization(name="batchnorm_conv5_1")(x)

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_2')(x)
    x = BatchNormalization(name="batchnorm_conv5_2")(x)

    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='conv5_3')(x)
    x = BatchNormalization(name="batchnorm_conv5_3")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, training=True)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, training=True)(x)
    x = Dense(classes, activation='softmax', name='predictions_vggd')(x)

    return Model([input_y, input_cbcr], x)
