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
            name='conv1_1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)


    concat = Concatenate(axis=-1)([x, norm_cbcr])
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(concat)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

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

    x = BatchNormalization(name="b_norm_64", input_shape=input_shape_y)(input_y)
    
    x = Conv2D(256, (3, 3),
            activation='relu',
            padding='same',
            name='conv1_1_dct_256')(x)
    # Block 1
    x = BatchNormalization(name="b_norm_192", input_shape=input_shape)(x)
    x = Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='conv1_1_dct_256')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='conv4_1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='conv4_2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    concat = Concatenate(axis=-1)([x, norm_cbcr])

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='conv5_1')(concat)
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
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs=[input_y, input_cbcr], outputs=x)


def VGG16A3CBNIDeconvolution(classes=1000):
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
    model.add(BatchNormalization(name="bn_1", input_shape=input_shape))
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1'))

    # Block '2'
    model.add(Conv2DTranspose(196, (3, 3), strides=(2, 2)))

    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16D3CBNIDeconvolution(classes=1000):
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
    model.add(BatchNormalization(name="bn_1", input_shape=input_shape))
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1'))

    # Block '2'
    model.add(Conv2DTranspose(196, (3, 3), strides=(2, 2)))

    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16A3CBNA(classes=1000):
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

    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1',
               input_shape=input_shape))

    model.add(BatchNormalization(name="b_norm_after"))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16D3CBNA(classes=1000):
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
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1',
               input_shape=input_shape))

    model.add(BatchNormalization(name="b_norm_after"))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16A3CBNADeconvolution(classes=1000):
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
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1',
               input_shape=input_shape))

    model.add(BatchNormalization(name="b_norm_after"))

    # Block '2'
    model.add(
        Conv2DTranspose(196, (3, 3), strides=(2, 2), name="block2_convt1"))

    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16D3CBNADeconvolution(classes=1000):
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
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1',
               input_shape=input_shape))

    model.add(BatchNormalization(name="b_norm_after"))

    # Block '2'
    model.add(
        Conv2DTranspose(196, (3, 3), strides=(2, 2), name="block2_convt1"))

    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2'))
    model.add(
        Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16A3CNoBN(classes=1000):
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
    model.add(
        Conv2D(196, (8, 8),
               strides=8,
               activation='relu',
               padding='same',
               name='block1_conv1',
               input_shape=input_shape))

    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1'))
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


def VGG16D64CBNI(classes=1000):
    """Instantiates the VGG16 architecture.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """

    # We have two inputs, Y/CBCr
    if K.image_data_format() == 'channels_last':
        input_shape_y = (28, 28, 64)
        input_shape_cbcr = (14, 14, 128)
    else:
        input_shape_y = (64, 28, 28)
        input_shape_cbcr = (14, 14, 128)

    img_input_y = Input(shape=input_shape_y, name="input_y")
    img_input_cbcr = Input(shape=input_shape_cbcr, name="input_cbcr")

    # Block 1
    x_y = BatchNormalization(name="bn_y")(img_input_y)
    x_cbcr = BatchNormalization(name="bn_cbcr")(img_input_cbcr)

    x_cbcr = Conv2DTranspose(128, (1, 1),
                             strides=(2, 2),
                             name="deconvolution_cbcr")(x_cbcr)

    x = Concatenate([x_y, x_cbcr], name="concatenate_components")

    # fit the input of the original next block
    x = Conv2D(128, (1, 1),
               strides=1,
               activation='relu',
               padding='same',
               name='block1_conv1_dct')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
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

    return Model([img_input_y, img_input_cbcr], x)