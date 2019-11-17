'''
A custom Keras layer to generate anchor boxes.

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
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf
from jpeg_deep.utils import convert_coordinates



class AnchorBoxesTensorflow(Layer):
    def __init__(self,
                 this_scale: float,
                 next_scale: float,
                 step,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
        '''
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                this_scale, next_scale))

        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.variances = np.array([0.1, 0.1, 0.2, 0.2])
        self.step = step
        self.n_boxes = len(self.aspect_ratios) + 1

        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale
                wh_list.append([box_width, box_height])
                # Compute one slightly larger version using the geometric mean of this scale value and the next.
                box_height = box_width = np.sqrt(
                    self.this_scale * self.next_scale)
                wh_list.append([box_width, box_height])
            else:
                box_height = self.this_scale / np.sqrt(ar)
                box_width = self.this_scale * np.sqrt(ar)
                wh_list.append([box_width, box_height])

        self.wh_list = wh_list

        self.tf_variances = tf.constant(
            self.variances, name="variances", dtype=tf.float32)
        self.tf_wh_list = tf.constant(
            self.wh_list, name="wh_list", dtype=tf.float32)
        self.tf_step = tf.constant(
            self.step, name="size_step", dtype=tf.float32)
        self.tf_n_boxes = tf.constant(self.n_boxes, name="n_boxes")

        super(AnchorBoxesTensorflow, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxesTensorflow, self).build(input_shape)

    def call(self, input_layer, mask=None):
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.

        batch_size = tf.shape(input_layer)[0]
        feature_map_height = tf.shape(input_layer)[1]
        feature_map_width = tf.shape(input_layer)[2]

        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cx = tf.linspace(0.5 * self.tf_step, (0.5 + tf.cast(feature_map_width,
                                                            tf.float32) - 1) * self.tf_step, feature_map_width)
        cy = tf.linspace(0.5 * self.tf_step, (0.5 + tf.cast(feature_map_height,
                                                            tf.float32) - 1) * self.tf_step, feature_map_height)

        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        cx_grid = tf.expand_dims(cx_grid, -1)
        cy_grid = tf.expand_dims(cy_grid, -1)

        cx_grid_b = tf.expand_dims(
            tf.tile(cx_grid, (1, 1, self.tf_n_boxes)), -1) / 300
        cy_grid_b = tf.expand_dims(
            tf.tile(cy_grid, (1, 1, self.tf_n_boxes)), -1) / 300

        wh_list_w = tf.expand_dims(tf.expand_dims(
            tf.expand_dims(self.tf_wh_list[:, 0], 0), 0), -1)
        wh_list_w = tf.tile(
            wh_list_w, (feature_map_height, feature_map_width, 1, 1)) / 300
        wh_list_h = tf.expand_dims(tf.expand_dims(
            tf.expand_dims(self.tf_wh_list[:, 1], 0), 0), -1)
        wh_list_h = tf.tile(
            wh_list_h, (feature_map_height, feature_map_width, 1, 1)) / 300

        boxes_tensor = tf.concat(
            [cx_grid_b, cy_grid_b, wh_list_w, wh_list_h], axis=-1)

        variances_tensor = tf.zeros_like(boxes_tensor, dtype=tf.float32)
        variances_tensor += self.tf_variances

        boxes_tensor_variances = tf.concat(
            [boxes_tensor, variances_tensor], axis=-1)

        boxes_tensor_variances = tf.expand_dims(boxes_tensor_variances, axis=0)
        boxes_tensor_variances = tf.tile(
            boxes_tensor_variances, (batch_size, 1, 1, 1, 1))

        return boxes_tensor_variances

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': self.aspect_ratios,
            'variances': list(self.variances),
            'n_boxes': self.n_boxes,
        }
        base_config = super(AnchorBoxesTensorflow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AnchorBoxes(Layer):
    '''
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 coords='centroids',
                 normalize_coords=True,
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'corners' for the format `(xmin, ymin, xmax,  ymax)`, or 'minmax' for the format `(xmin, xmax, ymin, ymax)`.
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        '''
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                this_scale, next_scale))

        self.img_height = 300
        self.img_width = 300
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.variances = np.array([0.1, 0.1, 0.2, 0.2])
        self.normalize_coords = normalize_coords
        self.n_boxes = len(self.aspect_ratios) + 1
        
        # Compute the number of boxes per cell
        if (1 in aspect_ratios):
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        The logic implemented here is identical to the logic in the module `ssd_box_encode_decode_utils.py`.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                
                box_height = box_width = np.sqrt(
                    self.this_scale * self.next_scale) * size
                wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor
        batch_size, feature_map_height, feature_map_width, _ = tf.keras.backend.int_shape(
            x)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height +
                                                       feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width +
                                                     feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want further down
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros(
            (feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(
            cx_grid, (1, 1, self.n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(
            cy_grid, (1, 1, self.n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= 300
            boxes_tensor[:, :, :, [1, 3]] /= 300


        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate(
            (boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(
            boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
