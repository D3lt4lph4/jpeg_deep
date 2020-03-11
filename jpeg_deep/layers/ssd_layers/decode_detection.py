'''
A custom Keras layer to decode the raw SSD prediction output. Corresponds to the
`DetectionOutput` layer type in the original Caffe implementation of SSD.
Copyright (C) 2018 Pierluigi Ferrari
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer


class DecodeDetections(Layer):
    '''
    A Keras layer to decode the raw SSD prediction output.
    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.
    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''
        All default argument values follow the Caffe implementation.
        Arguments:
            confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
                positive class in order to be considered for the non-maximum suppression stage for the respective class.
                A lower value will result in a larger part of the selection process being done by the non-maximum suppression
                stage, while a larger value will result in a larger part of the selection process happening in the confidence
                thresholding stage.
            iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
                with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
                to the box score.
            top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
                non-maximum suppression stage.
            nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
                suppression.
            coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
                i.e. the format `(cx, cy, w, h)` (box center coordinates, width, and height). Other coordinate formats are
                currently not supported.
            normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
                and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
                relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
                Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
                coordinates. Requires `img_height` and `img_width` if set to `True`.
            img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
            img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError(
                "This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        if coords != 'centroids':
            raise ValueError(
                "The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(
            self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(
            self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(
            self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(
            self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(
            self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(
            self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        '''

        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################

        # Convert anchor box offsets to image offsets.
        # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cx = y_pred[..., -12] * y_pred[..., -4] * \
            y_pred[..., -6] + y_pred[..., -8]
        # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * \
            y_pred[..., -5] + y_pred[..., -7]
        # w = exp(w_pred * variance_w) * w_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6]
        # h = exp(h_pred * variance_h) * h_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5]

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        # If the model predicts box coordinates relative to the image dimensions and they are supposed
        # to be converted back to absolute coordinates, do that.
        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(
            self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
        preds = tf.concat([xmin, ymin, xmax, ymax], axis=-1)
        scores = y_pred[..., 1:-12]

        preds = tf.expand_dims(preds, axis=2)

        f_pred = tf.image.combined_non_max_suppression(
            preds, scores, 400, 200, 0.45, 0.01, clip_boxes=False)

        conf = tf.expand_dims(f_pred[1], -1)
        classes = tf.expand_dims(f_pred[2], -1) + 1

        return tf.concat([classes, conf, f_pred[0]], axis=-1)
       
    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        # Last axis: (class_ID, confidence, 4 box coordinates)
        return (batch_size, self.tf_top_k, 6)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
