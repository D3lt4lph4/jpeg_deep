# Neural Networks using compressed JPEG images.

This repository provides code to train and used neural network on compressed JPEG images. The article describing the results is available [here](https://arxiv.org/abs/1904.08408).

This implementation relies on the module [jpeg2dct](https://github.com/uber-research/jpeg2dct) from uber research team. The SSD used in this repository was taken from [this repository](https://github.com/pierluigiferrari/ssd_keras) and then modified.

The following networks are available for usage in the repository:

- [VGG16](https://arxiv.org/abs/1409.1556)
- [ResNet50](https://arxiv.org/abs/1512.03385)
- [SSD](https://arxiv.org/abs/1512.02325)

For each of the classification networks, the fully convolutional part is available.
For each of the networks, the DCT version is also available.

#### Networks and limitations:

**Image Resizing**

Using the compressed representation of the data brings some limitations. The main limitation discussed here is the resizing of the input data. Resizing images in the RGB domain is straightforward whereas resizing in the DCT domain is more complicated. The following list of articles explore the possibility to resize images directly in the frequency domain:

- [On Resizing Images In The DCT Domain](https://ieeexplore.ieee.org/document/1421685)
- [Image Resizing In The Discrete Cosine Transform Domain](https://ieeexplore.ieee.org/document/537460)
- [Fast Image Resizing in Discrete Cosine Transform Domain with Spatial Relationship between DCT Block and its Sub-Blocks](https://ieeexplore.ieee.org/document/4590237)
- [Design and Analysis of an Image Resizing Filter in the Block-DCT Domain](https://www.researchgate.net/publication/3308607_Design_and_Analysis_of_an_Image_Resizing_Filter_in_the_Block-DCT_Domain)


These methods are so far not implemented in this repository. As the repository stands there are two way to overcome the resizing limitation. First, the networks in this repository were made (when possible) to be size independent. Any size images can be input to the networks. Although the inference works in theory, in practice there is a drop in performance. This drop is due to the fact that the size distribution of the objects in the images are changed when using images with a larger size (medium object become larger). So far the main recommendation is either to use fixed size images as input (when possible) or to improve the data-augmentation pipeline to follow the target objects size distribution.

**Training Pipeline**

The second limitation is for training, data-augmentation has to be carried in the RGB domain, thus the data-augmentation pipeline is the following one: JPEG => RGB => data-augmentation => JPEG => Compressed Input. This slows down the training.

## Installation

The installation steps are the same for classification and detection:

```bash
# Get the submodule for the templates
git submodule init
git submodule update

# Making virtualenv
mkdir .venv
cd .venv
python3 -m venv jpeg_deep
source jpeg_deep/bin/activate

cd ..

# Installation of the module
cd deep_template/template_keras/template_keras

python setup.py sdist
cd dist/
pip install template_keras-1.0.tar.gz

# Back to main directory
cd ../../../..

# Installing all the dependencies (the code was tested with the specified version numbers)
pip install keras
pip install tensorflow-gpu==1.14.0
pip install pillow
pip install opencv-python
pip install jpeg2dct
pip install albumentations
```

## Training

The training uses a system of config files and experiments. The aim is to help saving the parameters of a run.
Example config files are available in the config folder. The config files defines all the training and testing parameters. 

### System variables

To simplify deployment on different machines, the following variables need to be defined (see Classification and Detection section for details in the dataset_path):

```bash
# Setting the main dirs for the training datasets
export DATASET_PATH_TRAIN=<path_to_train_directory>
export DATASET_PATH_VAL=<path_to_validation_directory>
export DATASET_PATH_TEST=<path_to_test_directory>

# Setting the directory were the experiment folder will be created
export EXPERIMENTS_OUTPUT_DIRECTORY=<path_to_output_directory>

# Setting the directory were the logs will be output (optional, in case you use slurm)
export LOG_DIRECTORY=<path_to_log_directory>
```

### Starting the training

Once you have defined all the variables and modified the config files to your needs, simply run the following command:

```bash
python training.py -c <config_dir_path>
```

The config file in the <config_dir_path> needs to be named "config.py" for the script to run correctly.

### Training using horovod

The training script support the usage of horovod. An exemple file for training with horovod using slurm is provided [jpeg_deep.sl](slurm/jpeg_deep.sl).

```
cd slurm
sbatch jpeg_deep.sl
```

If you do not run on a multi-cluster computation facility that uses slurm, please refer to the original [horovod git](https://github.com/horovod/horovod)

## Inference

Before anything: **No pre-trained weights are/will be made available.** To get this repository running, you'll have to retrain the networks from scratch.

Inference can be done using the [inference.py](inference.py) script. In order to use the script you have to first carry a training for at least one epoch (the inference pre-suppose that you have an experiment folder).

The inference will be done on the test set. You need to modify the config_temp.py file in the experiment generated folder in order to use a different dataset.

```
python inference.py <experiment_path> <weights_path>
```

## Classification

The table below shows the results obtained compared with the state of the art. For the training, the ImageNet train set was used as training and the ImageNet validation set was used as validation. Except for the official results, the new results were evaluated reshaping the smallest size of the images to 256 and then center cropping.

| Newtork | top-1 | top-5 |
|:-:|:-:|:-:|
| VGG A (official) | 29.6 | 10.4 |
| VGG D (official) | 27.0 | 8.8 |
| VGG A (retrained) | 29.9 | 10.7 |
| VGG D (retrained) | 30.1 | 10.4 |
| VGG A DCT | 39.6 | 17.2 |
| VGG D DCT | 36.74 | 15.2 |

### Details in the dataset path

## Detection

### Details in the dataset path
