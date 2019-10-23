from typing import Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2

from jpeg_deep.layers.ssd_layers import AnchorBoxes, AnchorBoxesTensorflow, L2Normalization, DecodeDetections


# Helper functions
def identity_layer(tensor):
    return tensor


def input_mean_normalization(tensor):
    return tensor - np.array(subtract_mean)


def feature_map_rgb(image_shape: Tuple[int, int], l2_regularization: float = 0.0005, kernel_initializer: str = 'he_normal'):
    img_h, img_w = image_shape
    input_layer = Input(shape=(img_h, img_w, 3))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    lambda_layer = Lambda(identity_layer, output_shape=(img_h, img_w, 3),
                          name='identity_layer')(input_layer)
    input_mean_normalization = Lambda(input_mean_normalization, output_shape=(
        img_h, img_w, 3), name='input_mean_normalization')(lambda_layer)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='pool4')(conv4_3)
    return input_layer, pool4, conv4_3


def feature_map_dct(image_shape: Tuple[int, int], l2_regularization: float = 0.0005, kernel_initializer: str = 'he_normal'):

    if image_shape is None:
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

    conv1_1 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1_1_dct_256')(norm_y)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_1')(conv1_1)
    conv4_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_3)

    concat = Concatenate(axis=-1, name="")([pool4, norm_cbcr])

    return [input_y, input_cbcr], concat, conv4_3


def ssd300(n_classes: int,
           mode: str = 'training',
           kernel_initializer: str = 'he_normal',
           l2_regularization: float = 0.0005,
           confidence_thresh: float = 0.01,
           iou_threshold: float = 0.45,
           top_k: int = 200,
           nms_max_output_size: int = 400,
           dct: bool = False,
           image_shape: Tuple[int, int] = None):
    '''
    Builds a tensorflow.keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    # Arguments:
        - n_classes: The number of positive classes, e.g. 20 for Pascal VOC.
        - mode: One of 'training' or 'inference'.
        - kernel_initializer: The kernel to use to initialize all the layers.
        - l2_regularization: The L2-regularization rate. Applies to all convolutional layers. Set to zero to deactivate L2-regularization.
        - confidence_thresh: A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered for the non-maximum suppression stage for the respective class.
        - iou_threshold: A float in [0,1]. The IoU value above which the overlapping boxes will be removed.
        - top_k: The number of highest scoring predictions to be kept for each batch item after the non-maximum suppression stage.
        - nms_max_output_size: The maximal number of predictions that will be left over after the NMS stage.
        - image_shape: If known, the size of the inputs, in the format (height, width).


    # Returns:
        - model: The SSD model.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.

    scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]

    steps = [8, 16, 32, 64, 100, 300]
    size_steps = steps
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    n_boxes = [4, 6, 6, 6, 4, 4]

    subtract_mean = [123, 117, 104]
    swap_channels = [2, 1, 0]

    if mode == "training":
        img_h, img_w = 300, 300
    elif image_shape is None:
        img_h, img_w = None, None
    else:
        img_h, img_w = image_shape

    if not dct:
        input_layer, pool4, conv4_3 = feature_map_rgb(
            image_shape, l2_regularization=l2_regularization, kernel_initializer=kernel_initializer)
    else:
        input_layer, pool4, conv4_3 = feature_map_dct(
            image_shape, l2_regularization=l2_regularization, kernel_initializer=kernel_initializer)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(
        1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                 kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same',
                 kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv7_1')(conv6_2)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    # Build the convolutional predictor layers on top of the base network
    # We predict `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                                    kernel_regularizer=l2(l2_regularization), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same',
                           kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_regularization), name='conv9_2_mbox_conf')(conv9_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                                   kernel_regularizer=l2(l2_regularization), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(l2_regularization), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(l2_regularization), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(l2_regularization), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(l2_regularization), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(l2_regularization), name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor (prior) boxes
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)
    if image_shape is None:
        conv4_3_norm_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[0], next_scale=scales[1], step=size_steps[0], aspect_ratios=aspect_ratios[0], name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[1], next_scale=scales[2], step=size_steps[1], aspect_ratios=aspect_ratios[1], name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[2], next_scale=scales[3], step=size_steps[2], aspect_ratios=aspect_ratios[2], name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[3], next_scale=scales[4], step=size_steps[3], aspect_ratios=aspect_ratios[3], name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[4], next_scale=scales[5], step=size_steps[4], aspect_ratios=aspect_ratios[4], name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[5], next_scale=scales[6], step=size_steps[5], aspect_ratios=aspect_ratios[5], name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
    else:
        conv4_3_norm_mbox_priorbox = AnchorBoxes(
            img_h, img_w, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0], this_steps=steps[0], this_offsets=offsets[0])(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        this_steps=steps[1], this_offsets=offsets[1])(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                            this_steps=steps[2], this_offsets=offsets[2])(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                            this_steps=steps[3], this_offsets=offsets[3])(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                            this_steps=steps[4], this_offsets=offsets[4])(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                            this_steps=steps[5], this_offsets=offsets[5])(conv9_2_mbox_loc)

    # Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape(
        (-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape(
        (-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    # Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation(
        'softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(
        axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=input_layer, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               name='decoded_predictions',
                                               dct=dct)([predictions, input_layer])
        model = Model(inputs=input_layer, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    return model
