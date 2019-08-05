import unittest

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from vgg_jpeg_keras.generators import DCTGeneratorJPEG2DCT_111

class DCTGeneratorJPEG2DCT_111(unittest.TestCase):

    def test_generator_initialization(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json")

        self.assertTrue(len(generator.images_path) == 50000)
        self.assertTrue(len(generator.classes) == 1000)
        self.assertTrue(generator.batches_per_epoch == 1562)
        self.assertTrue(len(generator.indexes) == 50000)
        self.assertTrue(len(generator) == 1562)

    def test_get_item_no_scale(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json", scale=False)

        batch = generator[0]

        self.assertTrue(isinstance(batch, np.ndarray))
        self.assertTrue(batch.dtype == np.int32)
        self.assertTrue(batch[0].shape == (28, 28, 192))


    def test_get_item_no_scale(self):
        generator = DCTGeneratorJPEG2DCT_111("/save/2017018/bdegue01/datasets/imagenet/ILSVRC_2012/validation/", "data/imagenet_class_index.json", scale=False)

        batch = generator[0]

        self.assertTrue(isinstance(batch, np.ndarray))
        self.assertTrue(batch.dtype == np.int32)
        self.assertTrue(batch[0].shape == (28, 28, 192))
        self.assertTrue(batch[0,0,0,0:5].tolist() == [1,2,3,4,5])


if __name__ == '__main__':
    unittest.main()