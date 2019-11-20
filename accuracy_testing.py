from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

model = VGG16(weights='imagenet', include_top=True)

model.compile(optimizer=SGD(), loss=categorical_crossentropy, metrics=["accuracy"])

dir = "/data/thesis/datasets/imagenet/image_files/validation/"

generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(dir, target_size=(224,224), batch_size=2)

print(model.evaluate(generator))
