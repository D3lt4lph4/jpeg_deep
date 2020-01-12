from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

import numpy as np
from keras.metrics import top_k_categorical_accuracy

from keras.preprocessing.image import ImageDataGenerator

from jpeg_deep.generators import RGBGenerator
from jpeg_deep.networks import vggd, vggd_conv

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

gen = [SmallestMaxSize(256)]

def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    _func.__name__ = "_func_{}".format(k)
    return _func

model = vggd_conv(1000)
# model.load_weights(
#     "/home/benjamin/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
model.load_weights("/dlocal/home/2017018/bdegue01/weights/jpeg_deep/reproduce/vgg_conv.h5", by_name=True)
model.compile(optimizer=SGD(), loss=categorical_crossentropy,
              metrics=[_top_k_accuracy(1), _top_k_accuracy(5)])

dir = "/save/2017018/PARTAGE/imagenet/validation/"

generator = RGBGenerator(
    "/save/2017018/PARTAGE/imagenet/validation/", "/home/2017018/bdegue01/git/jpeg_deep/data/imagenet_class_index.json",input_size=(None), batch_size=1, transforms=gen)
print(model.evaluate_generator(generator, verbose=1))
