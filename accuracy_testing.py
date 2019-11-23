from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from jpeg_deep.generators import RGBGenerator
from jpeg_deep.networks import vggd

from keras.metrics import top_k_categorical_accuracy

from albumentations import (
    Blur,
    HorizontalFlip,
    RandomCrop,
    CenterCrop,
    RandomGamma,
    Rotate,
    OpticalDistortion,
    ElasticTransform,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    ChannelShuffle,
    SmallestMaxSize,
    RandomBrightnessContrast
)

from albumentations import (
    OneOf,
    Compose
)

gen = [SmallestMaxSize(256), CenterCrop(224, 224)]


def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    return _func


model = vggd(1000)
model.load_weights(
    "/home/benjamin/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
model.compile(optimizer=SGD(), loss=categorical_crossentropy,
              metrics=[_top_k_accuracy(1), _top_k_accuracy(5)])

dir = "/data/thesis/datasets/imagenet/image_files/validation/"

generator = RGBGenerator(
    "/d2/thesis/datasets/imagenet//validation", "./data/imagenet_class_index.json", batch_size=1, transforms=gen)

print(model.evaluate_generator(generator, verbose=1))
