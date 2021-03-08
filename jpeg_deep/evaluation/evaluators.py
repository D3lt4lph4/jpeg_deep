import sys
import numpy as np
import time
import json

from typing import List, Dict

from statistics import mean, stdev
from PIL import Image

from os import makedirs
from os.path import splitext, split, join

from tqdm import tqdm
from tqdm import trange

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from jpeg_deep.utils import iou

import tensorflow as tf


class Evaluator(object):
    def __init__(self, generator:object=None):
        """ General purpose evaluator. To be used when the calculation is run directly by keras.

        # Arguments:
            - generator: The generator to be used for the evaluation. May be None and specified later.
        """
        self.score = None
        self._generator = generator
        self.runs = False
        self.number_of_runs = None

    def __call__(self, model, test_generator=None, steps=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        self.score = model.evaluate_generator(self._generator, steps=steps, verbose=1)

    def model_speed(self, model:object, test_generator:object=None, number_of_runs:int=10, iteration_per_run:int=200, verbose:bool=True):
        """ Compute the speed of the network in FPS.

        # Arguments:
            - model: The model to use for prediction (keras model)
            - generator: The generator from which we will get the data (batch size of 1 only supported).
            - number_of_runs: The number of run to be done.
            - iteration_per_run: The number of batch predictions that will be done for each of the run.
            - verbose:Verbose mode of the function.
        
        # Returns:
            Nothing, will print information about the prediction speed of the network.

        """

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        it = tqdm(range(number_of_runs)) if verbose else range(number_of_runs)

        for _ in it:
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        """ Function to display more information about the prediction (graphs, ...).
        """
        print("The final score is:\n\t-loss: {:.02f}\n\t-top-1: {:.02f}\n\t-top-5: {:.02f}".format(*self.score))
        pass


class PascalEvaluator(object):
    def __init__(self, generator: object=None, n_classes:int=20, ignore_flagged_boxes:bool=True, challenge:str="VOC2007", set_type:str="test"):
        """ Evaluator for the Pascal VOC dataset.

        # Arguments:
            - generator: The generator that will provide with the data, should not be shuffled. Only supported with a batch size of 1 for now.
            - n_classes: The number of classes (without background).
            - ignore_flagged_boxes: If the flagged boxes should be ignored for evaluation (i.e difficult, truncated, ...)
            - challenge: The name of the challenge, will be used in case of generation of predictions for the submission of results.
            - set_type: The type of the set to use for evaluation (i.e train, val, test).

        """
        self.score = None
        self._generator = generator
        self.n_classes = n_classes
        self.challenge = challenge
        self.ignore_flagged_boxes = ignore_flagged_boxes
        self.set_type = set_type
        self.runs = False
        self.number_of_runs = None
        self.matching_iou_threshold = 0.5

    def __call__(self, model, test_generator=None, steps=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        # Get all the predictions
        self._generator.batch_size = 1
        self.shuffle = False
        generator_len = self._generator.number_of_data_samples

        images_path = self._generator.images_path

        results = [list() for _ in range(self.n_classes + 1)]

        for i in tqdm(range(generator_len)):
            X, y = self._generator.__getitem__(i)
            predictions = model.predict(X)
            image_id = splitext(split(images_path[i])[1])[0]

            for box in predictions[0]:
                class_id, confidence, xmin, ymin, xmax, ymax = box
                with Image.open(images_path[i]) as img:
                    width, height = img.size
                xmin = xmin * width / 300
                xmax = xmax * width / 300
                ymin = ymin * height / 300
                ymax = ymax * height / 300

                prediction = (image_id, confidence, xmin, ymin, xmax, ymax)

                results[int(class_id)].append(prediction)

        # Get the ground truth labels
        num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)
        ground_truth = self._generator.labels

        print('Computing the number of positive ground truth boxes per class.')
        tr = trange(len(ground_truth), file=sys.stdout)

        # Iterate over the ground truth for all images in the dataset.
        for i in tr:

            boxes = np.asarray(ground_truth[i])

            # Iterate over all ground truth boxes for the current image.
            for j in range(boxes.shape[0]):

                if self.ignore_flagged_boxes:
                    if not self._generator.flagged_boxes[i][j]:
                        # If this box is not supposed to be evaluation-neutral,
                        # increment the counter for the respective class ID.
                        class_id = boxes[j, 0]
                        num_gt_per_class[class_id] += 1
                else:
                    # If there is no such thing as evaluation-neutral boxes for
                    # our dataset, always increment the counter for the respective
                    # class ID.
                    class_id = boxes[j, 0]
                    num_gt_per_class[class_id] += 1

        # Match the prediction to the ground labels
        true_positives, false_positives, cumulative_true_positives, cumulative_false_positives = self.match_predictions(results,
                                                                                                                        ignore_neutral_boxes=True,
                                                                                                                        matching_iou_threshold=self.matching_iou_threshold,
                                                                                                                        border_pixels='include',
                                                                                                                        sorting_algorithm='quicksort')

        # Compute the cumulative precision and recall
        cumulative_precisions, cumulative_recalls = self.compute_precision_recall(
            cumulative_true_positives, cumulative_false_positives, num_gt_per_class)

        # Compute the average precision
        average_precisions = self.compute_average_precisions(
            cumulative_precisions, cumulative_recalls, mode="integrate", num_recall_points=11)

        # compute the mAP
        mean_average_precision = self.compute_mean_average_precision(
            average_precisions)

        print(average_precisions)
        print(mean_average_precision)

    def predict_for_submission(self, model: object, generator: object=None, output_dir:str="."):
        """ Function to create the file for submission on the evaluation servers.

        # Arguments:
            - model: The model to use for prediction (keras model)
            - generator: The generator from which we will get the data (batch size of 1 only supported).
            - output_dir: Where to output the results.
        
        # Returns:
            Nothing, a folder with the results will be created.
        
        """
        if self._generator is None and generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if generator is not None:
            self._generator = generator

        # First create the folders that are to hold the results
        full_output_dir = join(output_dir, "results", self.challenge, "Main")
        makedirs(full_output_dir)

        # Predicting for all the classes
        self._generator.batch_size = 1
        self.shuffle = False

        generator_len = self._generator.number_of_data_samples

        images_path = self._generator.images_path

        classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        results = [list() for _ in range(self.n_classes + 1)]

        for i in tqdm(range(generator_len)):
            X, y = self._generator.__getitem__(i)
            predictions = model.predict(X)
            image_id = splitext(split(images_path[i])[1])[0]

            for box in predictions[0]:
                class_id, confidence, xmin, ymin, xmax, ymax = box
                with Image.open(images_path[i]) as img:
                    width, height = img.size
                xmin = xmin * width / 300
                xmax = xmax * width / 300
                ymin = ymin * height / 300
                ymax = ymax * height / 300

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > width:
                    xmax = width
                if ymax > height:
                    ymax = height
                prediction = (image_id, confidence, float(xmin),
                              float(ymin), float(xmax), float(ymax))

                results[int(class_id)].append(prediction)

        # writing the predictions to the output folder
        for class_id in range(1, len(results)):
            output_file = join(
                full_output_dir, "comp3_det_{}_{}.txt".format(self.set_type, classes[class_id]))
            with open(output_file, "w") as class_file:
                for prediction in results[class_id]:
                    class_file.write(
                        "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*prediction))

    def model_speed(self, model: object, test_generator: object=None, number_of_runs: int=10, iteration_per_run:int=200, verbose:bool=False):
        """ Compute the speed of the network in FPS.

        # Arguments:
            - model: The model to use for prediction (keras model)
            - generator: The generator from which we will get the data (batch size of 1 only supported).
            - number_of_runs: The number of run to be done.
            - iteration_per_run: The number of batch predictions that will be done for each of the run.
            - verbose:Verbose mode of the function.
        
        # Returns:
            Nothing, will print information about the prediction speed of the network.

        """

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        it = tqdm(range(number_of_runs)) if verbose else range(number_of_runs)

        for _ in it:
            start_time = time.time()
            for _ in range(iteration_per_run):
                results = []
                pred = model.predict(X)

            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    @property
    def test_generator(self):
        return self._generator

    def match_predictions(self,
                          prediction_results:object,
                          ignore_neutral_boxes:bool=True,
                          matching_iou_threshold:float=0.5,
                          border_pixels:str='include',
                          sorting_algorithm:str='quicksort',
                          verbose:bool=True):
        '''
        Matches predictions to ground truth boxes.

        # Arguments:
            - prediction_results: Prediction results for an image.
            - ignore_neutral_boxes: If marked boxes should be ignored when matching prediction and groundtruth.
            - matching_iou_threshold: IoU threshold above which a groundtruth and a prediction are considered for matching.
            - border_pixels: How to treat the border pixels of the bounding boxes. Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong to the boxes. If 'exclude', the border pixels do not belong to the boxes. If 'half', then one of each of the two horizontal and vertical borders belong to the boxes, but not the other.
            - sorting_algorithm: Which sorting algorithm the matching algorithm should use. This argument accepts any valid sorting algorithm for Numpy's `argsort()` function.
            - verbose: If `True`, will print out the progress during runtime.
        
        # Returns:
            Four nested lists containing the true positives, false positives, cumulative true positives, and cumulative false positives for each class.
        '''

        class_id_gt = 0
        xmin_gt = 1
        ymin_gt = 2
        xmax_gt = 3
        ymax_gt = 4

        # Convert the ground truth to a more efficient format for what we need
        # to do, which is access ground truth by image ID repeatedly.
        ground_truth = {}
        # Whether or not we have annotations to decide whether ground truth boxes should be neutral or not.
        for i in range(len(self._generator.images_path)):
            image_id = splitext(split(self._generator.images_path[i])[1])[0]
            labels = self._generator.labels[i]
            if ignore_neutral_boxes:
                ground_truth[image_id] = (np.asarray(labels), np.asarray(
                    self._generator.flagged_boxes[i]))
            else:
                ground_truth[image_id] = np.asarray(labels)

        # The false positives for each class, sorted by descending confidence.
        true_positives = [[]]
        # The true positives for each class, sorted by descending confidence.
        false_positives = [[]]
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            predictions = prediction_results[class_id]

            # Store the matching results in these lists:
            # 1 for every prediction that is a true positive, 0 otherwise
            true_pos = np.zeros(len(predictions), dtype=np.int)
            # 1 for every prediction that is a false positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int)

            # In case there are no predictions at all for this class, we're done here.
            if len(predictions) == 0:
                print("No predictions for class {}/{}".format(class_id, self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                continue

            # Convert the predictions list for this class into a structured array so that we can sort it by confidence.

            # Get the number of characters needed to store the image ID strings in the structured array.
            # Keep a few characters buffer in case some image IDs are longer than others.
            num_chars_per_image_id = len(str(predictions[0][0])) + 6
            # Create the data type for the structured array.
            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            # Create the structured array
            predictions = np.array(predictions, dtype=preds_data_type)

            # Sort the detections by decreasing confidence.
            descending_indices = np.argsort(
                -predictions['confidence'], kind=sorting_algorithm)
            predictions_sorted = predictions[descending_indices]

            if verbose:
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description(
                    "Matching predictions to ground truth, class {}/{}.".format(class_id, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            # Keep track of which ground truth boxes were already matched to a detection.
            gt_matched = {}

            # Iterate over all predictions.
            for i in tr:

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                # Convert the structured array element to a regular array.
                pred_box = np.asarray(
                    list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))

                # Get the relevant ground truth boxes for this prediction,
                # i.e. all ground truth boxes that match the prediction's
                # image ID and class ID.

                # The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
                # or only `ground_truth_boxes`.
                if ignore_neutral_boxes:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)
                class_mask = gt[:, class_id_gt] == class_id
                gt = gt[class_mask]
                if ignore_neutral_boxes:
                    eval_neutral = eval_neutral[class_mask]

                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue

                # Compute the IoU of this prediction with all ground truth boxes of the same class.
                overlaps = iou(boxes1=gt[:, [xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)

                # For each detection, match the ground truth box with the highest overlap.
                # It's possible that the same ground truth box will be matched to multiple
                # detections.
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    # False positive, IoU threshold violated:
                    # Those predictions whose matched overlap is below the threshold become
                    # false positives.
                    false_pos[i] = 1
                else:
                    if not (ignore_neutral_boxes) or (eval_neutral[gt_match_index] == False):
                        # If this is not a ground truth that is supposed to be evaluation-neutral
                        # (i.e. should be skipped for the evaluation) or if we don't even have the
                        # concept of neutral boxes.
                        if not (image_id in gt_matched):
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(
                                shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                        elif not gt_matched[image_id][gt_match_index]:
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                        else:
                            # False positive, duplicate detection:
                            # If the matched ground truth box for this prediction has already been matched
                            # to a different prediction previously, it is a duplicate detection for an
                            # already detected object, which counts as a false positive.
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            # Cumulative sums of the true positives
            cumulative_true_pos = np.cumsum(true_pos)
            # Cumulative sums of the false positives
            cumulative_false_pos = np.cumsum(false_pos)

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

    def compute_precision_recall(self, cumulative_true_positives: object, cumulative_false_positives: object, num_gt_per_class: object, verbose: bool=True):
        '''
        Computes the cumulative precisions and recalls for all classes, i.e precision and recall at all the points.

        # Arguments:
            - cumulative_true_positives: The cumulative true positive as output by `match_predictions`
            - cumulative_false_positives: The cumulative false positive as output by `match_predictions`
            - num_gt_per_class: The number of groundtruth boxes per class.
            - verbose: If `True`, will print out the progress during runtime.

        # Returns:
            Two nested lists containing the cumulative precisions and recalls for each class.
        '''
    
        cumulative_precisions = [[]]
        cumulative_recalls = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print(
                    "Computing precisions and recalls, class {}/{}".format(class_id, self.n_classes))

            tp = cumulative_true_positives[class_id]
            fp = cumulative_false_positives[class_id]

            # 1D array with shape `(num_predictions,)`
            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
            # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / num_gt_per_class[class_id]

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        cumulative_precisions = cumulative_precisions
        cumulative_recalls = cumulative_recalls

        return cumulative_precisions, cumulative_recalls

    def compute_average_precisions(self, cumulative_precisions: List, cumulative_recalls: List, mode:str='sample', num_recall_points:int=11, verbose:bool=True):
        '''
        Computes the average precision for each class. Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling) and post-2010 (integration) algorithm versions.
    
        # Arguments:
            - cumulative_precisions: A nested list of the cumulative precisions values (one list per class).
            - cumulative_recalls: A nested list of the cumulative recall values (one list per class).
            - mode: Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision will be computed according to the Pascal VOC formula that was used up until VOC 2009. In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula that was used from VOC 2010 onward.
            - num_recall_points: Only relevant if mode is 'sample'. The number of points to sample from the precision-recall-curve to compute the average precisions. Eleven points is the value used in the official Pascal VOC pre-2010 detection evaluation algorithm.
            - verbose: If `True`, will print out the progress during runtime.

        # Returns:
            A list containing average precision for each class.
        '''
        average_precisions = [0.0]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print(
                    "Computing average precision, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = cumulative_precisions[class_id]
            cumulative_recall = cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                # For all the recall points
                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]
                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(
                    cumulative_recall, return_index=True, return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                # Iterate over all unique recall values in reverse order. This saves a lot of computation:
                # For each unique recall value `r`, we want to get the maximal precision value obtained
                # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
                # values after a given iteration, then in the next iteration, in order compute the maximal
                # precisions for the last `l > k` recall values, we only need to compute the maximal precision
                # for `l - k` recall values and then take the maximum between that and the previously computed
                # maximum instead of computing the maximum over all `l` values.
                # We skip the very last recall value, since the precision after between the last recall value
                # recall 1.0 is defined to be zero.
                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous iteration to
                    # avoid unnecessary repeated computation over the same precision values.
                    # The maximal precisions are the heights of the rectangle areas of our integral under
                    # the precision-recall curve.
                    maximal_precisions[i] = np.maximum(
                        np.amax(cumulative_precision[begin:end]), maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - \
                        unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        return average_precisions

    def compute_mean_average_precision(self, average_precisions):
        '''
        Computes the mean average precision over all classes.

        # Arguments:
            average_precisions: A list of all the average precision for each of the classes.
        
        # Returns:
            A float, the mean average precision.
        '''
        # The first element is for the background class, so skip it.
        mean_average_precision = np.average(average_precisions[1:])

        return mean_average_precision
    

    def display_results(self):
        """ Function to display more information about the prediction (graphs, ...).

        Not implemented.
        """
        pass


class CocoEvaluator(object):
    def __init__(self, annotation_file:str, generator:object=None, set:str="test-dev2017", network_name:str="dummy"):
        """ Evaluator for the MS-COCO dataset.

        # Arguments:
            - annotation_file: The file containing all the annotation for the images. Should be set even for prediction on the test dataset as it is used to extract information from the COCO api.
            - generator: The generator that will be used to run the predictions.
            - set: The name of the set that will be used for prediction. This argument is used to set the name of the results file in case of submission.
            - network_name: The name of the network that will be used for prediction. This argument is used to set the name of the results file in case of submission.

        """
        self.score = None
        self._generator = generator
        self.annotation_file = annotation_file
        self.runs = False
        self.number_of_runs = None
        self.set = set
        self.network_name = network_name

        # Getting the dictionnary matching the class/id
        self.coco = COCO(annotation_file)

        # display COCO categories and supercategories
        cats = self.coco.loadCats(self.coco.getCatIds())
        id_classes = sorted([(value["id"], value["name"])
                             for value in cats], key=lambda x: x[0])

        # add background class
        id_classes.insert(0, (0, "background"))

        # add the index for the predictions
        id_classes = [(value[0], value[1], i)
                      for i, value in enumerate(id_classes)]

        # create a dictionnary with the ids as keys
        self.matching_dictionnary = {value[2]: [
            value[0], value[1]] for value in id_classes}

    def __call__(self, model, test_generator=None, steps=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        # Get all the predictions
        self._generator.batch_size = 1
        self.shuffle = False
        generator_len = self._generator.number_of_data_samples

        images_path = self._generator.images_path

        results = []
        imgIds = []
        for i in tqdm(range(generator_len)):
            X, y = self._generator.__getitem__(i)

            predictions = model.predict(X)
            image_id = splitext(split(images_path[i])[1])[0]
            image_id = int(image_id.split("_")[-1])
            imgIds.append(image_id)

            for box in predictions[0]:
                class_id, confidence, xmin, ymin, xmax, ymax = box
                with Image.open(images_path[i]) as img:
                    width, height = img.size
                xmin = xmin * width / 300
                xmax = xmax * width / 300
                ymin = ymin * height / 300
                ymax = ymax * height / 300

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > width:
                    xmax = width
                if ymax > height:
                    ymax = height

                prediction = {"image_id": image_id, "category_id": self.matching_dictionnary[class_id][0], "bbox": [
                    xmin, ymin, xmax-xmin, ymax-ymin], "score": float(confidence)}

                results.append(prediction)

        with open("/tmp/output.json", "w") as file:
            json.dump(results, file)

        cocoGt = COCO(self.annotation_file)

        cocoDt = cocoGt.loadRes("/tmp/output.json")

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def predict_for_submission(self, model: object, generator: object=None, output_dir:str="."):
        """ Function to create the file for submission on the evaluation servers.

        # Arguments:
            - model: The model to use for prediction (keras model)
            - generator: The generator from which we will get the data (batch size of 1 only supported).
            - output_dir: Where to output the results.
        
        # Returns:
            Nothing, a folder with the results will be created.
        
        """
        if self._generator is None and generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        self.runs = False
        if generator is not None:
            self._generator = generator

        # Get all the predictions
        self._generator.batch_size = 1
        self.shuffle = False
        generator_len = self._generator.number_of_data_samples

        images_path = self._generator.images_path

        results = []
        imgIds = []
        for i in tqdm(range(generator_len)):
            X, y = self._generator.__getitem__(i)

            predictions = model.predict(X)
            image_id = splitext(split(images_path[i])[1])[0]
            image_id = int(image_id.split("_")[-1])
            imgIds.append(image_id)

            for box in predictions[0]:
                class_id, confidence, xmin, ymin, xmax, ymax = box
                with Image.open(images_path[i]) as img:
                    width, height = img.size
                xmin = xmin * width / 300
                xmax = xmax * width / 300
                ymin = ymin * height / 300
                ymax = ymax * height / 300

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > width:
                    xmax = width
                if ymax > height:
                    ymax = height

                prediction = {"image_id": image_id, "category_id": self.matching_dictionnary[class_id][0], "bbox": [
                    xmin, ymin, xmax-xmin, ymax-ymin], "score": float(confidence)}

                results.append(prediction)

        output_file = join(output_dir, "{}_{}_{}.json".format(
            "detections", self.set, self.network_name))
        with open(output_file, "w") as file_json:
            json.dump(results, file_json)

    def model_speed(self, model:object, test_generator:object=None, number_of_runs:int=10, iteration_per_run:int=200, verbose:bool=False):
        """ Compute the speed of the network in FPS.

        # Arguments:
            - model: The model to use for prediction (keras model)
            - generator: The generator from which we will get the data (batch size of 1 only supported).
            - number_of_runs: The number of run to be done.
            - iteration_per_run: The number of batch predictions that will be done for each of the run.
            - verbose:Verbose mode of the function.
        
        # Returns:
            Nothing, will print information about the prediction speed of the network.

        """

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        it = tqdm(range(number_of_runs)) if verbose else range(number_of_runs)

        for _ in tqdm(range(number_of_runs)):
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        """ Function to display more information about the prediction (graphs, ...).

        Not implemented.
        """
        pass
