import unittest
import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np

from vgg_jpeg_keras.generators import DCTGeneratorJPEG2DCT_111, prepare_imagenet

class test_DCTGeneratorJPEG2DCT_111(unittest.TestCase):

    def test_generator_initialization(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json")

        self.assertTrue(len(generator.images_path) == 50000)
        self.assertTrue(len(generator.classes) == 1000)
        self.assertTrue(generator.batches_per_epoch == 1562)
        self.assertTrue(len(generator.indexes) == 50000)
        self.assertTrue(len(generator) == 1562)

    def test_get_item_scale_default(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json", shuffle=False)

        batch, y = generator[0]
        
        self.assertTrue(len(batch) == 2)
        self.assertTrue(len(batch[0]) == len(y))
        self.assertTrue(len(batch[1]) == len(y))

        self.assertTrue(isinstance(batch, list))
        self.assertTrue(batch[0].dtype == np.int32)
        self.assertTrue(batch[1].dtype == np.int32)
        self.assertTrue(batch[0][0].shape == (28, 28, 64))
        self.assertTrue(batch[1][0].shape == (14, 14, 128))

        self.assertTrue(isinstance(y, np.ndarray))
        self.assertTrue(y.dtype == np.int32)

        self.assertTrue(y[0, 462] == 1)

    def test_get_item_no_scale_default(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json", scale=False, shuffle=False)

        batch, y = generator[0]
        
        self.assertTrue(len(batch) == 2)
        self.assertTrue(len(batch[0]) == len(y))
        self.assertTrue(len(batch[1]) == len(y))

        self.assertTrue(len(batch[0]) == 32)

        self.assertTrue(isinstance(batch, list))
        self.assertTrue(batch[0].dtype == np.int32)
        self.assertTrue(batch[1].dtype == np.int32)
        self.assertTrue(batch[0][0].shape == (28, 28, 64))
        self.assertTrue(batch[1][0].shape == (14, 14, 128))
        
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertTrue(y.dtype == np.int32)

        self.assertTrue(batch[0][0,0,0,0:5].tolist() == [-616, -24, 10, 0, -12])
        self.assertTrue(batch[1][0,0,0,0:5].tolist() == [0, 0, 0, 0, 0])
        self.assertTrue(y[0, 462] == 1)
        
    def test_prepare_imagenet(self):
        index_file = "data/imagenet_class_index.json"
        data_directory = "/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/"
        association, classes, images_path = prepare_imagenet(index_file, data_directory)

        self.assertTrue(len(classes) == 1000)
        print(association.keys())
        self.assertTrue(len(association.keys()) == 1000)
        self.assertTrue(len(images_path) == 50000)



if __name__ == '__main__':
    unittest.main()