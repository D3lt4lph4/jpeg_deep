import numpy as np

import matplotlib.pyplot as plt

import json

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
        # Extract the relevent information from the index file.
        self.classes = {}
        with open(index_file) as index:
            data = json.load(index)
            for key in data:
                self.classes[int(key)] = data[key][1]
        

    def display(self, predictions: object, inputs: object):
        """ Function to display the predictions on top of the input image.

        # Arguments:
            - predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, 1000).
            - inputs: The images input to the classifier.
        
        # Returns:
            Nothing, will display all the predictions.
        """

        # Iterate over the predictions
        for k in range(len(predictions)):

            # Image to display
            img = inputs[k]

            # Get the best prediction
            idx = np.argmax(predictions[k])

            # Write the prediction with the confidence on the image
            label = '{}: {:.2f}'.format(self.classes[int(idx)], predictions[k][idx])

            plt.figure(label)
            plt.imshow(img)
            plt.show()

    def display_with_gt(self, predictions: object, inputs: object, groundtruth: object):
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

            # Image to display
            img = inputs[k]

            # Get the best prediction
            idx_pred = np.argmax(predictions[k])
            # Get the actual label
            idx_gt = np.argmax(groundtruth[k])

            # Write the prediction with the confidence on the image
            label = 'Predicted: {} ({:.2f})'.format(self.classes[int(idx_pred)], predictions[k][idx_pred])
            gt = 'Groundtruth: {}'.format(self.classes[int(idx_gt)])
            
            plt.figure("Image {}".format(k + 1))
            plt.text(0,-30, gt)
            plt.text(0,-15, label)
            plt.imshow(img)
            plt.show()

class DisplayerObjects(object):
    def __init__(self, classes: List[str]=None, confidence_threshold: float=0.5):
        """ Displayer for object detection tasks.

        # Arguments:
            - classes: A list of all the classes to be predicted, should be in the same order as the labels. If None, uses the Pascal VOC classes.
            - confidence_threshold: The threshold under which objects will not be considered as positive detection.
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
        """ Displays the predictions on top of the input image.

        # Arguments:
            - predictions: The predictions as returned by the detection network, should be an array of size (batch_size, n_boxes, 6). Last dim is (prediction, confidence, x_min, y_min, x_max, y_max).
            - inputs: The inputs to the detector, should have batch size len.
        
        # Returns:
            Nothing, will display all the predictions.
        """
        # Filter out the predictions with confidence lower than threshold
        y_pred_thresh = [predictions[k][predictions[k, :, 1] >
                                        self.confidence_threshold] for k in range(predictions.shape[0])]

        # For each of the images in the batch
        for k in range(len(predictions)):
            # Display the image as background
            plt.figure(figsize=(20, 12))
            plt.imshow(inputs[k])

            # Get the axis to add the boxes
            current_axis = plt.gca()

            for box in y_pred_thresh[k]:
                # Get the coordinates of the box
                xmin, ymin, xmax, ymax = box[2:]

                # Set the boxes in the limit of the input image
                if xmin < 0:
                    xmin = 0

                if ymin < 0:
                    ymin = 0

                if xmax > inputs[k].shape[1]:
                    xmax = inputs[k].shape[1]

                if ymax > inputs[k].shape[0]:
                    ymax = inputs[k].shape[0]

                # Display the prediction
                color = self.colors[int(box[0])]
                label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle(
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large',
                                  color='white', bbox={'facecolor': color, 'alpha': 1.0})
            # Show the image
            plt.show()

    def display_with_gt(self, predictions: object, inputs: object, groundtruth: object):
        """ Function to display the predictions on top of the input image. The ground truth will be added (Not implemented).

        # Arguments:
            - predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, n_boxes, 6). Last dim is  (prediction, confidence, x_min, y_min, x_max, y_max).
            - inputs: The inputs images to the classifier.
            - groundtruth: The real label of the predictions, should be an array of size (batch_size, n_boxes, 6). Last dim is (prediction, confidence, x_min, y_min, x_max, y_max).
        
        # Returns:
            Nothing, will display all the predictions alongside the groundtruths.
        """
        # Filter out the predictions with confidence lower than threshold
        y_pred_thresh = [predictions[k][predictions[k, :, 1] >
                                        self.confidence_threshold] for k in range(predictions.shape[0])]

        # For each of the images in the batch
        for k in range(len(predictions)):
            # Display the images as background
            f, axarr = plt.subplots(1,2,figsize=(15,15))

            axarr[0].set_title("Predicted")
            axarr[0].imshow(inputs[k])
            axarr[1].set_title("Groundtruth")
            axarr[1].imshow(inputs[k])

            for box in groundtruth[k]:
                # Get the coordinates of the box
                xmin, ymin, xmax, ymax = box[1:]

                # Display the box
                color = self.colors[int(box[0])]
                label = '{}'.format(self.classes[int(box[0])])
                axarr[1].add_patch(plt.Rectangle(
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                axarr[1].text(xmin, ymin, label, size='x-large',
                                  color='white', bbox={'facecolor': color, 'alpha': 1.0})

            for box in y_pred_thresh[k]:
                # Get the coordinates of the box
                xmin, ymin, xmax, ymax = box[2:]

                # Set the boxes in the limit of the input image
                if xmin < 0:
                    xmin = 0

                if ymin < 0:
                    ymin = 0

                if xmax > inputs[k].shape[1]:
                    xmax = inputs[k].shape[1]

                if ymax > inputs[k].shape[0]:
                    ymax = inputs[k].shape[0]

                # Display the prediction
                color = self.colors[int(box[0])]
                label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
                axarr[0].add_patch(plt.Rectangle(
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                axarr[0].text(xmin, ymin, label, size='x-large',
                                  color='white', bbox={'facecolor': color, 'alpha': 1.0})
            # Show the image
            plt.show()
