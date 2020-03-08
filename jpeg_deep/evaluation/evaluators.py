from template_keras.evaluators import TemplateEvaluator
import sys
import numpy as np
import time
import json
from statistics import mean, stdev
from PIL import Image

from os import makedirs
from os.path import splitext, split, join

from tqdm import tqdm
from tqdm import trange

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from jpeg_deep.utils import iou


class Evaluator(TemplateEvaluator):
    def __init__(self, generator=None):
        self.score = None
        self._generator = generator
        self.runs = False
        self.number_of_runs = None

    def __call__(self, model, test_generator=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        self.score = model.evaluate_generator(self._generator, verbose=1)

    def model_speed(self, model, test_generator=None, number_of_runs=10, iteration_per_run=200):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        for _ in tqdm(range(number_of_runs)):
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        scores = []
        self.runs = True

        if test_generator is not None:
            self._generator = test_generator

        if test_generator is not None:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))
        else:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))

        self.score = np.mean(np.array(scores), axis=0)
        self.number_of_runs = number_of_runs

    def __str__(self):
        if self.runs:
            return "Number of runs: {}\nAverage score: {}".format(
                self.number_of_runs, self.score)
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        print("The evaluated score is {}.".format(self.score))


class PascalEvaluator(TemplateEvaluator):
    def __init__(self, generator=None, n_classes=20, ignore_flagged_boxes=True, challenge="VOC2007", set_type="test"):
        self.score = None
        self._generator = generator
        self.n_classes = n_classes
        self.challenge = challenge
        self.ignore_flagged_boxes = ignore_flagged_boxes
        self.set_type = set_type
        self.runs = False
        self.number_of_runs = None
        self.matching_iou_threshold = 0.5

    def __call__(self, model, test_generator=None):
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
            average_precisions, ret=True)

        print(average_precisions)
        print(mean_average_precision)

    def predict_for_submission(self, model, generator=None, output_dir="."):
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

    def model_speed(self, model, test_generator=None, number_of_runs=10, iteration_per_run=1000):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        for _ in tqdm(range(number_of_runs)):
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        scores = []
        self.runs = True

        if test_generator is not None:
            self._generator = test_generator

        if test_generator is not None:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))
        else:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))

        self.score = np.mean(np.array(scores), axis=0)
        self.number_of_runs = number_of_runs

    def __str__(self):
        if self.runs:
            return "Number of runs: {}\nAverage score: {}".format(
                self.number_of_runs, self.score)
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        print("The evaluated score is {}.".format(self.score))

    def match_predictions(self,
                          prediction_results,
                          ignore_neutral_boxes=True,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True):
        '''
        Matches predictions to ground truth boxes.
        Note that `predict_on_dataset()` must be called before calling this method.
        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the true and false positives.
        Returns:
            None by default. Optionally, four nested lists containing the true positives, false positives, cumulative true positives,
            and cumulative false positives for each class.
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

    def compute_precision_recall(self, ctp, cfp, ngpc, verbose=True, ret=True):
        '''
        Computes the precisions and recalls for all classes.
        Note that `match_predictions()` must be called before calling this method.
        Arguments:
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the precisions and recalls.
        Returns:
            None by default. Optionally, two nested lists containing the cumulative precisions and recalls for each class.
        '''
        cumulative_true_positives = ctp
        cumulative_false_positives = cfp
        num_gt_per_class = ngpc
        if (cumulative_true_positives is None) or (cumulative_false_positives is None):
            raise ValueError(
                "True and false positives not available. You must run `match_predictions()` before you call this method.")

        if (num_gt_per_class is None):
            raise ValueError(
                "Number of ground truth boxes per class not available. You must run `get_num_gt_per_class()` before you call this method.")

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

    def compute_average_precisions(self, cp, cr, mode='sample', num_recall_points=11, verbose=True, ret=True):
        '''
        Computes the average precision for each class.
        Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
        and post-2010 (integration) algorithm versions.
        Note that `compute_precision_recall()` must be called before calling this method.
        Arguments:
            mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision will be computed
                according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled for `num_recall_points`
                recall values. In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula that
                was used from VOC 2010 onward, where the average precision will be computed by numerically integrating over the whole
                preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just the limit case
                of 'sample' mode as the number of sample points increases. For details, see the references below.
            num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of points to sample from the precision-recall-curve
                to compute the average precisions. In other words, this is the number of equidistant recall values for which the resulting
                precision will be computed. 11 points is the value used in the official Pascal VOC pre-2010 detection evaluation algorithm.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the average precisions.
        Returns:
            None by default. Optionally, a list containing average precision for each class.
        References:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
        '''
        cumulative_precisions = cp
        cumulative_recalls = cr

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

    def compute_mean_average_precision(self, ap, ret=True):
        '''
        Computes the mean average precision over all classes.
        Note that `compute_average_precisions()` must be called before calling this method.
        Arguments:
            ret (bool, optional): If `True`, returns the mean average precision.
        Returns:
            A float, the mean average precision, by default. Optionally, None.
        '''
        average_precisions = ap

        # The first element is for the background class, so skip it.
        mean_average_precision = np.average(average_precisions[1:])

        return mean_average_precision


class CocoEvaluator(TemplateEvaluator):
    def __init__(self, generator=None, n_classes=80, ignore_flagged_boxes=True, challenge="VOC2007", set_type="test"):
        self.score = None
        self._generator = generator
        self.n_classes = n_classes
        self.challenge = challenge
        self.set_type = set_type
        self.runs = False
        self.number_of_runs = None

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


    def __call__(self, model, test_generator=None):
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

        for i in tqdm(range(generator_len)):
            X, y = self._generator.__getitem__(i)

            predictions = model.predict(X)
            image_id = int(splitext(split(images_path[i])[1])[0])

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
                
                prediction = {"image_id": image_id,"category_id": self.matching_dictionnary[class_id],"bbox": [xmin, ymin, xmax-xmin, ymax-ymin],"score":float(confidence)}

                results.append(prediction)
    
    with open("/tmp/output.json", "w") as file:
        json.dump(results, file)
    
    dataDir='/d2/thesis/datasets/mscoco'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    annType = "bbox"     #specify type here
    prefix = 'instances'

    cocoGt=COCO(annFile)

    cocoDt=cocoGt.loadRes("/tmp/output.json")

    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    def predict_for_submission(self, model, generator=None, output_dir="."):
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

    def model_speed(self, model, test_generator=None, number_of_runs=10, iteration_per_run=1000):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        for _ in tqdm(range(number_of_runs)):
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        scores = []
        self.runs = True

        if test_generator is not None:
            self._generator = test_generator

        if test_generator is not None:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))
        else:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))

        self.score = np.mean(np.array(scores), axis=0)
        self.number_of_runs = number_of_runs

    def __str__(self):
        if self.runs:
            return "Number of runs: {}\nAverage score: {}".format(
                self.number_of_runs, self.score)
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        print("The evaluated score is {}.".format(self.score))
