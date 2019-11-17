import numpy as np
from matplotlib import pyplot as plt
from template_keras.displayers import TemplateDisplayer

import matplotlib
matplotlib.use("Qt5Agg")


class Displayer(TemplateDisplayer):

    def __init__(self):
        pass

    def display(self, predictions, inputs):
        pass

    def display_with_gt(self, predictions, inputs, groundtruth):
        pass


class DisplayerObjects(TemplateDisplayer):

    def __init__(self, classes=None, confidence_threshold=0.5):
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
        self.colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    def display(self, predictions, inputs):
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

    def display_with_gt(self, predictions, inputs, groundtruth):
        pass
