#import matplotlib.pyplot as plt

# from template.displayer import TemplateDisplayer

class Displayer(object):

    def __init__(self):
        pass

    # def display_with_gt(self, predictions, inputs, groundtruth):
    #     for i, gt in enumerate(groundtruth):
    #         plt.title('Ground Truth : {}\nPrediction: {}'.format(gt.argmax(), predictions[i].argmax()))
    #         plt.imshow(inputs[i][:,:,0])
    #         plt.show()

    # def display(self, predictions, inputs):
    #     for i, prediction in enumerate(predictions):
    #         plt.title('Prediction: {}'.format(prediction.argmax()))
    #         plt.imshow(inputs[i][:,:,0])
    #         plt.show()
