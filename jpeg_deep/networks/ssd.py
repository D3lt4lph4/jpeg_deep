from typing import Tuple

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate, BatchNormalization, ZeroPadding2D
from keras.regularizers import l2

from jpeg_deep.layers.ssd_layers import AnchorBoxes, AnchorBoxesTensorflow, L2Normalization, DecodeDetections


# Helper functions
def identity_layer(tensor):
    return tensor


def input_channel_swap(tensor):
    swap_channels = [2, 1, 0]
    return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)


def input_mean_normalization(tensor):
    return tensor - np.array([123, 117, 104])


def feature_map_rgb(image_shape: Tuple[int, int], kernel_initializer: str = 'glorot_uniform'):
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

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    lambda_layer = Lambda(identity_layer, output_shape=(img_h, img_w, 3),
                          name='identity_layer')(input_layer)
    input_mean_normalization_layer = Lambda(input_mean_normalization, output_shape=(
        img_h, img_w, 3), name='input_mean_normalization')(lambda_layer)

    input_swap = Lambda(input_channel_swap, output_shape=(
        img_h, img_w, 3), name='input_channel_swap')(input_mean_normalization_layer)

    block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block1_conv1')(input_mean_normalization_layer)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block1_pool')(block1_conv2)

    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block2_pool')(block2_conv2)

    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block3_pool')(block3_conv3)

    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding='same', name='block4_pool')(block4_conv3)
    return input_layer, block4_pool, block4_conv3


def feature_map_dct(image_shape: Tuple[int, int],  kernel_initializer: str = 'glorot_uniform'):
    """ Helper function that generates the first layers of the SSD. This function generates the layers for the DCT network.

    # Arguments:
        - image_shape: A tuple containing the shape of the image.
        - l2_regularization: The float value for the l2 normalization.
        - kernel_initializer: The type of initializer for the convolution kernels.

    # Returns:
        Three layers: input_layer, block4_pool, block4_conv3. These layers are used to intantiate the network.
    """
    if image_shape[0] is None:
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

    block1_conv1 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1_dct_256')(norm_y)

    # Block 4
    block4_conv1 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1_dct')(block1_conv1)
    block4_conv2 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool')(block4_conv3)

    concat = Concatenate(axis=-1, name="concat_dct")([block4_pool, norm_cbcr])

    return [input_y, input_cbcr], concat, block4_conv3


def SSD300(n_classes: int = 20,
           mode: str = 'training',
           kernel_initializer: str = 'glorot_uniform',

           confidence_thresh: float = 0.01,
           iou_threshold: float = 0.45,
           top_k: int = 200,
           nms_max_output_size: int = 400,
           dct: bool = False,
           image_shape: Tuple[int, int] = (300, 300)):
    '''
    Builds a ssd network, the network built can either be an RGB or DCT network. For more details on the architecture, see [the article](https://arxiv.org/abs/1512.02325v5).

    # Arguments:
        - n_classes: The number of positive classes, e.g. 20 for Pascal VOC.
        - mode: One of 'training' or 'inference'.
        - kernel_initializer: The kernel to use to initialize all the layers.
        - l2_regularization: The L2-regularization rate. Applies to all convolutional layers. Set to zero to deactivate L2-regularization.
        - confidence_thresh: A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered for the non-maximum suppression stage for the respective class.
        - iou_threshold: A float in [0,1]. The IoU value above which the overlapping boxes will be removed.
        - top_k: The number of highest scoring predictions to be kept for each batch item after the non-maximum suppression stage.
        - nms_max_output_size: The maximal number of predictions that will be left over after the NMS stage.
        - dct: A boolean to set the network for DCT inference (or RGB is False).
        - image_shape: If known, the size of the inputs, in the format (height, width).


    # Returns:
        - model: A keras model, representation of the SSD.
    '''
    n_classes += 1  # Account for the background class.

    #  Scales values for the anchor boxes
    scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]

    # The size of a step between the cells.
    steps = [8, 16, 32, 64, 100, 300]

    # The offsets for the boxes.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # The number of anchor boxes for the given output layer.
    n_boxes = [4, 6, 6, 6, 4, 4]

    two_boxes_for_ar1 = True

    clip_boxes = False

    coords = 'centroids'

    normalize_coords = True

    # The variances by which the encoded target coordinates are divided as in the original implementation
    variances = [0.1, 0.1, 0.2, 0.2]

    # If we train we set to the pre-set size else either None or the specified size.
    if mode == "training":
        img_h, img_w = 300, 300
    elif image_shape is None:
        img_h, img_w = None, None
    else:
        img_h, img_w = image_shape

    # Prepare the feature extractor.
    if not dct:
        input_layer, block4_pool, block4_conv3 = feature_map_rgb(
            image_shape, kernel_initializer=kernel_initializer)
    else:
        input_layer, block4_pool, block4_conv3 = feature_map_dct(
            image_shape, kernel_initializer=kernel_initializer)

    # Create the network.
    if dct:
        name = 'block5_conv1_dct'
    else:
        name = 'block5_conv1'
    block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name=name)(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D(pool_size=(3, 3), strides=(
        1, 1), padding='same', name='block5_pool')(block5_conv3)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                 kernel_initializer=kernel_initializer, name='fc6')(block5_pool)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same',
                 kernel_initializer=kernel_initializer, name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, name='conv9_2')(conv9_1)

    # Feed block4_conv3 into the L2 normalization layer
    block4_conv3_norm = L2Normalization(
        gamma_init=20, name='block4_conv3_norm')(block4_conv3)

    # Build the convolutional predictor layers on top of the base network
    # We predict `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    block4_conv3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                                         name='block4_conv3_norm_mbox_conf')(block4_conv3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same',
                           kernel_initializer=kernel_initializer, name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same',
                               kernel_initializer=kernel_initializer, name='conv9_2_mbox_conf')(conv9_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    block4_conv3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                                        name='block4_conv3_norm_mbox_loc')(block4_conv3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                              name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor (prior) boxes
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)
    if image_shape[0] is None:
        block4_conv3_norm_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[0], next_scale=scales[1], step=steps[0], aspect_ratios=aspect_ratios[0], name='block4_conv3_norm_mbox_priorbox')(block4_conv3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[1], next_scale=scales[2], step=steps[1], aspect_ratios=aspect_ratios[1], name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[2], next_scale=scales[3], step=steps[2], aspect_ratios=aspect_ratios[2], name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[3], next_scale=scales[4], step=steps[3], aspect_ratios=aspect_ratios[3], name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[4], next_scale=scales[5], step=steps[4], aspect_ratios=aspect_ratios[4], name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxesTensorflow(
            this_scale=scales[5], next_scale=scales[6], step=steps[5], aspect_ratios=aspect_ratios[5], name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
    else:
        block4_conv3_norm_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                                      two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
            0], this_offsets=offsets[0], clip_boxes=clip_boxes,
            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(block4_conv3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                            1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                                2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                                3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                                4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                                5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

        # block4_conv3_norm_mbox_priorbox = AnchorBoxes(
        #     img_h, img_w, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0], this_steps=steps[0], this_offsets=offsets[0])(block4_conv3_norm_mbox_loc)
        # fc7_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
        #                                 this_steps=steps[1], this_offsets=offsets[1])(fc7_mbox_loc)
        # conv6_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
        #                                     this_steps=steps[2], this_offsets=offsets[2])(conv6_2_mbox_loc)
        # conv7_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
        #                                     this_steps=steps[3], this_offsets=offsets[3])(conv7_2_mbox_loc)
        # conv8_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
        #                                     this_steps=steps[4], this_offsets=offsets[4])(conv8_2_mbox_loc)
        # conv9_2_mbox_priorbox = AnchorBoxes(img_h, img_w, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
        #                                     this_steps=steps[5], this_offsets=offsets[5])(conv9_2_mbox_loc)

    # Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    block4_conv3_norm_mbox_conf_reshape = Reshape(
        (-1, n_classes), name='block4_conv3_norm_mbox_conf_reshape')(block4_conv3_norm_mbox_conf)
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
    block4_conv3_norm_mbox_loc_reshape = Reshape(
        (-1, 4), name='block4_conv3_norm_mbox_loc_reshape')(block4_conv3_norm_mbox_loc)
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
    block4_conv3_norm_mbox_priorbox_reshape = Reshape(
        (-1, 8), name='block4_conv3_norm_mbox_priorbox_reshape')(block4_conv3_norm_mbox_priorbox)
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
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([block4_conv3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([block4_conv3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([block4_conv3_norm_mbox_priorbox_reshape,
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
        axis=-1, name='predictions_ssd')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

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
            "`mode` must be one of 'training' or 'inference', but received '{}'.".format(mode))

    return model
