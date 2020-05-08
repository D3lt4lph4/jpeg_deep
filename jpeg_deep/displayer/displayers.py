import json

import numpy as np

import matplotlib.pyplot as plt

from typing import List

class ImageNetDisplayer(object):
    def __init__(self, index_file: str):
        """ Displayer for an imagenet based classifier.

        # Argument:
            - index_file: The json file containing the mapping of the index/classes (see details below).
        
        # Details:

        Details of the content of the index file:
        ```json
        {
            "0": [
                "n01440764",
                "tench"],
            ...
        }
        ```

        """
        # Transform the dictionary on int based values
        self.classes = {}
        with open(index_file) as index:
            data = json.load(index)
            for key in data:
                self.classes[int(key)] = data[key][1]
        

    def display(self, predictions: object, inputs: object):
        """ Function to display the predictions on top of the input image.

        # Arguments:
            - predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, 1000)
            - inputs: The inputs images to the classifier.
        
        # Returns:
            Nothing, will display all the predictions.
        """

        # Iterate over the predictions
        for k in range(len(predictions)):

            # Display the image
            img = inputs[k]

            # Get the best prediction
            idx = np.argmax(predictions[k])

            # Write the prediction with the confidence on the image
            label = '{}: {:.2f}'.format(self.classes[int(idx)], predictions[k][idx])

            plt.figure(label)
            plt.imshow(img)
            plt.show()

    def display_with_gt(self, prediction: object, inputs: object, groundtruth: object):
        """ Function to display the predictions on top of the input image. The ground truth will be added.

        # Arguments:
            - predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, 1000).
            - inputs: The inputs images to the classifier.
            - groundtruth: The real label of the predictions, should be an array of size (batch_size, 1000).
        
        # Returns:
            Nothing, will display all the predictions alongside the groundtruths.
        """
        # Iterate over the predictions
        for k in range(len(predictions)):

            # Display the image
            img = inputs[k]

            # Get the best prediction
            idx_pred = np.argmax(predictions[k])
            idx_gt = np.argmax(groundtruth[k])

            # Write the prediction with the confidence on the image
            label = 'Predicted: {}, {:.2f}'.format(self.classes[int(idx_pred)], predictions[k][idx_pred])
            gt = 'Groundtruth: {}'.format(self.classes[int(idx_gt)])
            
            plt.figure("Image {}".format(k + 1))
            plt.text(0,-30, gt)
            plt.text(0,-15, label)
            plt.imshow(img)
            plt.show()

class DisplayerObjects(object):
    def __init__(self, classes: List[str]=None, confidence_threshold: float=0.5):
        """ Displayer detection tasks.

        # Argument:
            - classes: A list of all the classes to be predicted, should be in the same order as the labels. If None, will use the Pascal VOC classes.
            - confidence_threshold: The threshold under which objects will not be considered as detection.
        """
        if classes is None:
            self.classes = ['background',
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat',
                            'chair', 'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']
        else:
            self.classes = classes

        self.confidence_threshold = confidence_threshold
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes))).tolist()

    def display(self, predictions: object, inputs: object):
        """ Function to display the predictions on top of the input image.

        # Arguments:
            - predictions: The predictions as returned by the detection network, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).
            - inputs: The inputs images to the detector.
        
        # Returns:
            Nothing, will display all the predictions.
        """
        y_pred_thresh = [predictions[k][predictions[k, :, 1] >
                                        self.confidence_threshold] for k in range(predictions.shape[0])]
        for k in range(len(predictions)):

            # Set the colors for the bounding boxes
            plt.figure(figsize=(20, 12))
            plt.imshow(inputs[k])

            current_axis = plt.gca()

            print(y_pred_thresh[k])

            for box in y_pred_thresh[k]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[2]  # * original_images[k].shape[1]
                ymin = box[3]  # * original_images[k].shape[0]
                xmax = box[4]  # * original_images[k].shape[1]
                ymax = box[5]  # * original_images[k].shape[0]

                if xmin < 0:
                    xmin = 0

                if ymin < 0:
                    ymin = 0

                if xmax > inputs[k].shape[1]:
                    xmax = inputs[k].shape[1]

                if ymax > inputs[k].shape[0]:
                    ymax = inputs[k].shape[0]

                color = self.colors[int(box[0])]
                label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle(
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large',
                                  color='white', bbox={'facecolor': color, 'alpha': 1.0})

            plt.show()

    def display_with_gt(self, predictions: object, inputs: object, groundtruth: object):
        """ Function to display the predictions on top of the input image. The ground truth will be added.

        # Arguments:
            - predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).
            - inputs: The inputs images to the classifier.
            - groundtruth: The real label of the predictions, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).
        
        # Returns:
            Nothing, will display all the predictions alongside the groundtruths.
        """
        pass
