'''
An encoder that converts ground truth annotations to SSD-compatible training targets.

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

import numpy as np

from typing import List

from jpeg_deep.utils import iou, convert_coordinates
from jpeg_deep.utils import iou


def match_bipartite_greedy(weight_matrix):
    '''
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''

    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    # Only relevant for fancy-indexing below.
    all_gt_indices = list(range(num_ground_truth_boxes))

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):

        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        # Reduce along the anchor box axis.
        anchor_indices = np.argmax(weight_matrix, axis=1)
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        # Reduce along the ground truth box axis.
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches


def match_multi(weight_matrix, threshold):
    '''
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    '''

    num_anchor_boxes = weight_matrix.shape[1]
    # Only relevant for fancy-indexing below.
    all_anchor_indices = list(range(num_anchor_boxes))

    # Find the best ground truth match for every anchor box.
    # Array of shape (weight_matrix.shape[1],)
    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met


class BoundGenerator:
    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):
        '''
        Generates pairs of floating point values that represent lower and upper bounds
        from a given sample space.

        # Arguments:
            - sample_space (list or tuple): A list, tuple, or array-like object of shape
                `(n, 2)` that contains `n` samples to choose from, where each sample
                is a 2-tuple of scalars and/or `None` values.
            - weights (list or tuple, optional): A list or tuple representing the distribution
                over the sample space. If `None`, a uniform distribution will be assumed.
        '''

        if (not (weights is None)) and len(weights) != len(sample_space):
            raise ValueError(
                "`weights` must either be `None` for uniform distribution or have the same length as `sample_space`.")

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError(
                    "All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None:
                bound_pair[0] = 0.0
            if bound_pair[1] is None:
                bound_pair[1] = 1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError(
                    "For all sample space elements, the lower bound cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0/self.sample_space_size] * \
                self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        '''
        Returns:
            An item of the sample space, i.e. a 2-tuple of scalars.
        '''
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]


class BoxFilter:
    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1,
                                'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Returns all bounding boxes that are valid with respect to a the defined criteria.

        # Arguments:
            - check_overlap (bool, optional): Whether or not to enforce the overlap requirements defined by
                `overlap_criterion` and `overlap_bounds`. Sometimes you might want to use the box filter only
                to enforce a certain minimum area for all boxes (see next argument), in such cases you can
                turn the overlap requirements off.
            - check_min_area (bool, optional): Whether or not to enforce the minimum area requirement defined
                by `min_area`. If `True`, any boxes that have an area (in pixels) that is smaller than `min_area`
                will be removed from the labels of an image. Bounding boxes below a certain area aren't useful
                training examples. An object that takes up only, say, 5 pixels in an image is probably not
                recognizable anymore, neither for a human, nor for an object detection model. It makes sense
                to remove such boxes.
            - check_degenerate (bool, optional): Whether or not to check for and remove degenerate bounding boxes.
                Degenerate bounding boxes are boxes that have `xmax <= xmin` and/or `ymax <= ymin`. In particular,
                boxes with a width and/or height of zero are degenerate. It is obviously important to filter out
                such boxes, so you should only set this option to `False` if you are certain that degenerate
                boxes are not possible in your data and processing chain.
            - overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within the given `overlap_bounds`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within the given `overlap_bounds`.
            - overlap_bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            - min_area (int, optional): Only relevant if `check_min_area` is `True`. Defines the minimum area in
                pixels that a bounding box must have in order to be valid. Boxes with an area smaller than this
                will be removed.
            - labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            - border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        '''
        if not isinstance(overlap_bounds, (list, tuple, BoundGenerator)):
            raise ValueError(
                "`overlap_bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(overlap_bounds, (list, tuple)) and (overlap_bounds[0] > overlap_bounds[1]):
            raise ValueError(
                "The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError(
                "`overlap_criterion` must be one of 'iou', 'area', or 'center_point'.")
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        '''
        Arguments:
            labels (array): The labels to be filtered. This is an array with shape `(m,n)`, where
                `m` is the number of bounding boxes and `n` is the number of elements that defines
                each bounding box (box coordinates, class ID, etc.). The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): Only relevant if `check_overlap == True`. The height of the image
                (in pixels) to compare the box coordinates to.
            image_width (int): `check_overlap == True`. The width of the image (in pixels) to compare
                the box coordinates to.

        Returns:
            An array containing the labels of all boxes that are valid.
        '''

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Record the boxes that pass all checks here.
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:

            non_degenerate = (
                labels[:, xmax] > labels[:, xmin]) * (labels[:, ymax] > labels[:, ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:

            min_area_met = (labels[:, xmax] - labels[:, xmin]) * \
                (labels[:, ymax] - labels[:, ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:

            # Get the lower and upper bounds.
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # Compute which boxes are valid.

            if self.overlap_criterion == 'iou':
                # Compute the patch coordinates.
                image_coords = np.array([0, 0, image_width, image_height])
                # Compute the IoU between the patch and all of the ground truth boxes.
                image_boxes_iou = iou(image_coords, labels[:, [
                                      xmin, ymin, xmax, ymax]], coords='corners', mode='element-wise', border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou > lower) * \
                    (image_boxes_iou <= upper)

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
                    d = 1
                elif self.border_pixels == 'exclude':
                    # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
                    d = -1
                # Compute the areas of the boxes.
                box_areas = (labels[:, xmax] - labels[:, xmin] + d) * \
                    (labels[:, ymax] - labels[:, ymin] + d)
                # Compute the intersection area between the patch and all of the ground truth boxes.
                clipped_boxes = np.copy(labels)
                clipped_boxes[:, [ymin, ymax]] = np.clip(
                    labels[:, [ymin, ymax]], a_min=0, a_max=image_height-1)
                clipped_boxes[:, [xmin, xmax]] = np.clip(
                    labels[:, [xmin, xmax]], a_min=0, a_max=image_width-1)
                # +1 because the border pixels belong to the box areas.
                intersection_areas = (clipped_boxes[:, xmax] - clipped_boxes[:, xmin] + d) * (
                    clipped_boxes[:, ymax] - clipped_boxes[:, ymin] + d)
                # Check which boxes meet the overlap requirements.
                if lower == 0.0:
                    # If `self.lower == 0`, we want to make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
                    mask_lower = intersection_areas > lower * box_areas
                else:
                    # Especially for the case `self.lower == 1` we want the ">=" sign, otherwise no boxes would count at all.
                    mask_lower = intersection_areas >= lower * box_areas
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # Compute the center points of the boxes.
                cy = (labels[:, ymin] + labels[:, ymax]) / 2
                cx = (labels[:, xmin] + labels[:, xmax]) / 2
                # Check which of the boxes have center points within the cropped patch remove those that don't.
                requirements_met *= (cy >= 0.0) * (cy <= image_height-1) * \
                    (cx >= 0.0) * (cx <= image_width-1)

        return labels[requirements_met]


class ImageValidator:
    '''
    
    '''

    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 n_boxes_min=1,
                 labels_format={'class_id': 0, 'xmin': 1,
                                'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Returns `True` if a given minimum number of bounding boxes meets given overlap
        requirements with an image of a given height and width.

        # Arguments:
            - overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within `lower` and `upper`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within `lower` and `upper`.
            - bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            - n_boxes_min (int or str, optional): Either a non-negative integer or the string 'all'.
                Determines the minimum number of boxes that must meet the `overlap_criterion` with respect to
                an image of the given height and width in order for the image to be a valid image.
                If set to 'all', an image is considered valid if all given boxes meet the `overlap_criterion`.
            - labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            - border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        '''
        if not ((isinstance(n_boxes_min, int) and n_boxes_min > 0) or n_boxes_min == 'all'):
            raise ValueError(
                "`n_boxes_min` must be a positive integer or 'all'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format
        self.border_pixels = border_pixels
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds,
                                    labels_format=self.labels_format,
                                    border_pixels=self.border_pixels)

    def __call__(self,
                 labels,
                 image_height,
                 image_width):
        '''
        Arguments:
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): The height of the image to compare the box coordinates to.
            image_width (int): The width of the image to compare the box coordinates to.

        Returns:
            A boolean indicating whether an imgae of the given height and width is
            valid with respect to the given bounding boxes.
        '''

        self.box_filter.overlap_bounds = self.bounds
        self.box_filter.labels_format = self.labels_format

        # Get all boxes that meet the overlap requirements.
        valid_labels = self.box_filter(labels=labels,
                                       image_height=image_height,
                                       image_width=image_width)

        # Check whether enough boxes meet the requirements.
        if isinstance(self.n_boxes_min, int):
            # The image is valid if at least `self.n_boxes_min` ground truth boxes meet the requirements.
            if len(valid_labels) >= self.n_boxes_min:
                return True
            else:
                return False
        elif self.n_boxes_min == 'all':
            # The image is valid if all ground truth boxes meet the requirements.
            if len(valid_labels) == len(labels):
                return True
            else:
                return False


class SSDInputEncoder:
    def __init__(self,
                 img_height=300,
                 img_width=300,
                 n_classes=20,
                 predictor_sizes=None,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.5,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):
        '''
        Transforms ground truth labels for object detection in images
        (2D bounding box coordinates and class labels) to the format required for
        training an SSD model.

        In the process of encoding the ground truth labels, a template of anchor boxes
        is being built, which are subsequently matched to the ground truth boxes
        via an intersection-over-union threshold criterion.

        # Arguments:
            - img_height (int): The height of the input images.
            - img_width (int): The width of the input images.
            - n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            - predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            - min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be >0.
            - max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last predictor layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if `two_boxes_for_ar1` is `True`. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be greater than or equal to `min_scale`.
            - scales (list, optional): A list of floats >0 containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first `k` elements are the
                scaling factors for the `k` predictor layers, while the last element is used for the second box
                for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
                last scaling factor must be passed either way, even if it is not being used. If a list is passed,
                this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
                Note that you should set the scaling factors such that the resulting anchor box sizes correspond to
                the sizes of the objects you are trying to detect.
            - aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Note that you should set the aspect ratios such
                that the resulting anchor box shapes roughly correspond to the shapes of the objects you are trying to detect.
            - aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Note that you should set the aspect ratios such
                that the resulting anchor box shapes very roughly correspond to the shapes of the objects you are trying to detect.
            - two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            - steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
                either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
                pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
                the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
                If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
                If no steps are provided, then they will be computed such that the anchor box center points will form an
                equidistant grid within the image dimensions.
            - offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
                either floats or tuples of two floats. These numbers represent for each predictor layer how many
                pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
                as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
                of the step size specified in the `steps` argument. If the list contains floats, then that value will
                be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
                `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
            - clip_boxes (bool, optional): If `True`, limits the anchor box coordinates to stay within image boundaries.
            - variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            - matching_type (str, optional): Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth box will
                be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in addition to the aforementioned
                bipartite matching, all anchor boxes with an IoU overlap greater than or equal to the `pos_iou_threshold` will be
                matched to a given ground truth box.
            - pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box.
            - neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            - border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            - coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
                and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            - normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be within [0,1].
                This way learning becomes independent of the input image size.
            - background_id (int, optional): Determines which class ID is for the background class.
        '''
        ##################################################################################
        # Handle exceptions.
        ##################################################################################

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError(
                "Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if len(variances) != 4:
            raise ValueError(
                "4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError(
                "All variances must be >0, but the variances given are {}".format(variances))

        if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
            raise ValueError(
                "Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError(
                "You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError(
                "You must provide at least one offset value per predictor layer.")

        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1  # + 1 for the background class

        if predictor_sizes is None:
            self.predictor_sizes = np.array([
                (38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)])
        else:
            self.predictor_sizes = np.array(predictor_sizes)
        if self.predictor_sizes.ndim == 1:
            self.predictor_sizes = np.expand_dims(self.predictor_sizes, axis=0)

        self.min_scale = min_scale
        self.max_scale = max_scale
        # If `scales` is None, compute the scaling factors by linearly interpolating between
        # `min_scale` and `max_scale`. If an explicit list of `scales` is given, however,
        # then it takes precedent over `min_scale` and `max_scale`.
        if (scales is None):
            self.scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        else:
            # If a list of scales is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`.
            self.scales = scales
        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers. If `aspect_ratios_per_layer` is given,
        # however, then it takes precedent over `aspect_ratios_global`.
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [[1.0, 2.0, 0.5],
                                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                  [1.0, 2.0, 0.5],
                                  [1.0, 2.0, 0.5]]
        else:
            # If aspect ratios are given per layer, we'll use those.
            self.aspect_ratios = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [8, 16, 32, 64, 100, 300]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        # Compute the number of boxes per spatial location for each predictor layer.
        # For example, if a predictor layer has three different aspect ratios, [1.0, 0.5, 2.0], and is
        # supposed to predict two boxes of slightly different size for aspect ratio 1.0, then that predictor
        # layer predicts a total of four boxes at every spatial location across the feature map.
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################

        # Compute the anchor boxes for each predictor layer. We only have to do this once
        # since the anchor boxes depend only on the model configuration, not on the input data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.

        # This will store the anchor boxes for each predicotr layer.
        self.boxes_list = []

        # The following lists just store diagnostic information. Sometimes it's handy to have the
        # boxes' center points, heights, widths, etc. in a list.
        self.wh_list_diag = []  # Box widths and heights for each predictor layer
        # Horizontal and vertical distances between any two boxes for each predictor layer
        self.steps_diag = []
        self.offsets_diag = []  # Offsets for each predictor layer
        # Anchor box center points as `(cy, cx)` for each predictor layer
        self.centers_diag = []

        # Iterate over all predictor layers and compute the anchor boxes for each one.
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        '''
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''

        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        ##################################################################################
        # Generate the template for y_encoded.
        ##################################################################################

        y_encoded = self.generate_encoding_template(
            batch_size=batch_size, diagnostics=False)

        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################

        # Match the ground truth boxes to the anchor boxes. Every anchor box that does not have
        # a ground truth match and for which the maximal IoU overlap with any ground truth box is less
        # than or equal to `neg_iou_limit` will be a negative (background) box.

        # All boxes are background boxes by default.
        y_encoded[:, :, self.background_id] = 1
        # The total number of boxes that the model predicts per batch item
        n_boxes = y_encoded.shape[1]
        # An identity matrix that we'll use as one-hot class vectors
        class_vectors = np.eye(self.n_classes)

        for i in range(batch_size):  # For each batch item...

            if ground_truth_labels[i].size == 0:
                # If there is no ground truth for this batch item, there is nothing to match.
                continue
            labels = ground_truth_labels[i].astype(
                np.float)  # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                # Normalize ymin and ymax relative to the image height
                labels[:, [ymin, ymax]] /= self.img_height
                # Normalize xmin and xmax relative to the image width
                labels[:, [xmin, xmax]] /= self.img_width

            # Maybe convert the box coordinate format.
            if self.coords == 'centroids':
                labels = convert_coordinates(
                    labels, start_index=xmin, conversion='corners2centroids', border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(
                    labels, start_index=xmin, conversion='corners2minmax')

            # The one-hot class IDs for the ground truth boxes of this batch item
            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
            # The one-hot version of the labels for this batch item
            labels_one_hot = np.concatenate(
                [classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)

            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12:-8],
                               coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)

            # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            #        This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            bipartite_matches = match_bipartite_greedy(
                weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #         such ground truth box.

            if self.matching_type == 'multi':

                # Get all matches that satisfy the IoU threshold.
                matches = match_multi(
                    weight_matrix=similarities, threshold=self.pos_iou_threshold)

                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # Third: Now after the matching is done, all negative (background) anchor boxes that have
            #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
            #        i.e. they will no longer be background boxes. These anchors are "too close" to a
            #        ground truth box to be valid background boxes.

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(
                max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################

        if self.coords == 'centroids':
            # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [-12, -11]] -= y_encoded[:, :, [-8, -7]]
            # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:, :, [-12, -11]] /= y_encoded[:,
                                                     :, [-6, -5]] * y_encoded[:, :, [-4, -3]]
            # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= y_encoded[:, :, [-6, -5]]
            # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
            y_encoded[:, :, [-10, -9]
                      ] = np.log(y_encoded[:, :, [-10, -9]]) / y_encoded[:, :, [-2, -1]]
        elif self.coords == 'corners':
            # (gt - anchor) for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -10]] /= np.expand_dims(
                y_encoded[:, :, -6] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-11, -9]] /= np.expand_dims(
                y_encoded[:, :, -5] - y_encoded[:, :, -7], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]
        elif self.coords == 'minmax':
            # (gt - anchor) for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(
                y_encoded[:, :, -7] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(
                y_encoded[:, :, -5] - y_encoded[:, :, -6], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            # Keeping the anchor box coordinates means setting the offsets to zero.
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        '''
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        # Arguments:
            - feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            - aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            - this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            - next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            - diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        # Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(
                        this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height +
                                                       feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width +
                                                     feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want further down
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros(
            (feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(
            boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the SSD model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        # Arguments:
            - batch_size (int): The batch size.
            - diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        # Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros(
            (batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encoding_template = np.concatenate(
            (classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template


class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
