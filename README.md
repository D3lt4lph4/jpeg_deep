# Neural Networks using compressed JPEG images.

This repository provides code to train and used neural network on compressed JPEG images. **No pre-trained weights are/will be made available.** The article for this repository is available [here](https://arxiv.org/abs/2006.05732).

This implementation relies on the module [jpeg2dct](https://github.com/uber-research/jpeg2dct) from uber research team. The SSD used in this repository was taken from [this repository](https://github.com/pierluigiferrari/ssd_keras) and then modified.

All the networks proposed in this repository are modified versions of the three following architectures

- [VGG16](https://arxiv.org/abs/1409.1556)
- [ResNet50](https://arxiv.org/abs/1512.03385)
- [SSD](https://arxiv.org/abs/1512.02325)

## Summary

1. [Installation](#Installation])
2. [Training](#Training)
3. [Prediction](#Predict)
4. [Classification (ImageNet)](#Classification-ImageNet)
5. [Detection (PascalVOC)](#Detection-(PascalVOC))
6. [Detection (MS-COCO)](#Detection-(MS-COCO))
7. [Method limitations](#Method-limitations)

## Installation

The provided code can be used directly or install as a package. The following steps are to install the dependencies in a virtual env:

```bash
# Making virtualenv
mkdir .venv
cd .venv
python3 -m venv jpeg_deep
source jpeg_deep/bin/activate

cd ..

# Installing all the dependencies (the code was tested with the specified version numbers on python 3.+)
pip install keras
pip install tensorflow-gpu==1.14.0
pip install pillow
pip install opencv-python
pip install jpeg2dct
pip install albumentations
pip install tqdm
pip install bs4
pip install cython
pip install pycocotools
pip install matplotlib
pip install lxml
```

## Training

The training uses a system of configuration files and experiments. This system aims to help saving the parameters of a given run. On start of the training, an experiment folder will be created with copies of the configuration files, weights and logs. Examples config files available are the configuration used for the different training of the [paper](https://arxiv.org/abs/2006.05732).

### System variables

To simplify deployment on different machines, the following variables need to be defined (see the Classification/Detections sections for details in the dataset_path):

```bash
# Setting the main dirs for the training datasets
export DATASET_PATH_TRAIN=<path_to_train_directory>
export DATASET_PATH_VAL=<path_to_validation_directory>
export DATASET_PATH_TEST=<path_to_test_directory>

# Setting the directory were the experiment folder will be created
export EXPERIMENTS_OUTPUT_DIRECTORY=<path_to_output_directory>
```

### Starting the training

Once you have defined all the variables and modified the config files to your needs, simply run the following command (be aware that using horovod requires the modification of some of the parameters):

```bash
python scripts/training.py -c <config_dir_path> --no-horovod
```

The config file in the <config_dir_path> needs to be named "config.py" for the script to run correctly.

For more details on classification training on ImageNet dataset, refer to this [section](#Classification-ImageNet), for more details for training on Pascal VOC dataset, refer to this [section](#Detection-(Pascal-VOC)) and for more details for training MS-COCO dataset, refer to this [section](#Detection-(MS-COCO))

### Training using horovod

The training script support the usage of horovod. Be aware that using horovod requires the modification of some of the parameters, I recommend reading these articles for more details on multi gpu training: [1](https://arxiv.org/abs/1404.5997), [2](https://www.research.ed.ac.uk/portal/files/75846467/width_of_minima_reached_by_stochastic_gradient_descent_is_influenced_by_learning_rate_to_batch_size_ratio.pdf) and [3](https://arxiv.org/abs/1706.02677).

I highly recommend to train on multiple GPUs for the classification on ImageNet given the size of the dataset. An exemple file for training with horovod using slurm is provided [jpeg_deep.sl](slurm/jpeg_deep.sl).

```bash
cd slurm
sbatch jpeg_deep.sl
```

This script is given as example and will probably not work for your settings, you can refer to the original [horovod git](https://github.com/horovod/horovod) for more details on how to get it to work.

## Predict

**No pre-trained weights are/will be made available.** To get this section running, you'll have to retrain the networks from scratch.

### Display the results

Displaying the results can be done using the [prediction.py](scripts/prediction.py) script. In order to use the script you have to first carry a training for at least one epoch (the prediction pre-suppose that you have an experiment folder).

The prediction will be done on the test set. You need to modify the config_temp.py file in the experiment generated folder in order to use a different dataset.

**For the vgg16 based classifiers:** The prediction script uses the test generator specified in the config file to get the data. Hence, with the provided examples, you may need first to convert the weights to a fully convolutional version of the network. This can be done using the [classification2ssd.py](scripts/classification2ssd.py) script:

```bash
# Create the .h5 for vgg16
python scripts/classification2ssd.py vgg16 <weights_path>

# Create the .h5 for vgg_dct
python scripts/classification2ssd.py vgg16 <weights_path> -dct
```

As the ResNet is fully convolutional, the script above needs not to be run on the ResNet weights.

Once this is done, simply run the following command:

```bash
python scripts/prediction.py <experiment_path> <weights_path>
```

### Prediction time

We also provide with a way to test the speed of the trained networks. This is done using the [prediction_time.py](scripts/prediction_time.py) script.

In order to test the speed of the networks, a batch of data is preloaded into memory then prediction is run over this batch for P times, and the overall is done N times. Results is then averaged. You may or may not load weights.

```bash
python scripts/prediction_time.py <experiment_path> -nr 10 -w <weights_path>
```

### Evaluation

The trained networks can be evaluated using the following script:

```bash
python scripts/evaluate.py <experiment_path> <weights_path>
```

This script can also generate file for submission on the evaluation servers for the PascalVOC and MSCOCO:

```bash
python scripts/evaluate.py <experiment_path> <weights_path> -s -o <output_path>
```

## Classification (ImageNet)

### Results on ImageNet

The table below shows the results obtained (accuracy) compared with the state of the art. All the presented results are on the validation dataset. All the FPS were calculated using a NVIDIA GTX 1080 and using the [prediction_time.py](scripts/prediction_time.py) script. Batch size was set to 8.

| Official Newtorks | top-1 | top-5 | FPS |
|:-|:-:|:-:|:-:|
| [VGG16](https://arxiv.org/abs/1409.1556) | 73.0 | 91.2 | N/A |
| [VGG-DCT](https://arxiv.org/abs/1904.08408) | 42.0 | 66.9 | N/A |
| [ResNet50](https://arxiv.org/abs/1904.08408) | 75.78 | 92.65 | N/A |
| [LC-RFA](https://arxiv.org/abs/1904.08408) | 75.92 | 92.81 | N/A |
| [LC-RFA-Thinner](https://arxiv.org/abs/1904.08408) | 75.39 | 92.57 | N/A |
| [Deconvolution-RFA](https://arxiv.org/abs/1904.08408) | 76.06 | 92.02 | N/A |

| VGG based Newtorks (our trainings) | top-1 | top-5 | FPS |
|:-|:-:|:-:|:-:|
| VGG16 | 71.9 | 90.8 | 267 |
| VGG-DCT | 65.5 | 86.4 | 553 |
| VGG-DCT Y | 62.6 | 84.6 | 583 |
| VGG-DCT Deconvolution | 65.9 | 86.7 | 571 |

| ResNet50 based Newtorks (our trainings) | top-1 | top-5 | FPS |
|:-|:-:|:-:|:-:|
| ResNet50 | 74.73 | 92.33 | 324 |
| LC-RFA | **74.82** | **92.58** | 318 |
| LC-RFA Y | 73.25 | 91.40 | 329 |
| LC-RFA-Thinner | 74.62 | 92.33 | 389 |
| LC-RFA-Thinner Y | 72.48 | 91.04 | 395 |
| Deconvolution-RFA | 74.55 | 92.39 | 313 |

### Training on ImageNet

The dataset can be downloaded [here](https://academictorrents.com/browse.php?search=imagenet). Choose the version that suits your needs, I used the 2012 (Object Detection) data.

Once the data is downloaded, to use the provided generators, it should be stored following this tree (as long as you have separated train and validation folders you should be okay)

```text
imagenet
|
|_ train
|  |_ n01440764
|  |_ n01443537
|  |_ ...
|
|_ validation
   |_ n01440764
   |_ n01443537
   |_ ...
```

Then you'll just need to set the configuration files to fit your needs and follow the procedure described in the [training](##Training) section. **Keep in mind that the provided configuration files were used in a distributed training, hence the hyper parameters fit this particular settings. If you don't train that way, you'll need to change them.**

Also the system variable should be set to the ImageNet folder (if you use the provided config files)

```bash
# Setting the main dirs for the training datasets
export DATASET_PATH_TRAIN=<path_to_train_directory>/imagenet
export DATASET_PATH_VAL=<path_to_validation_directory>/imagenet
export DATASET_PATH_TEST=<path_to_test_directory>/imagenet
```

## Detection (Pascal VOC)

### Results on the PASCAL VOC dataset

Results for training on the Pascal VOC dataset are presented bellow. Networks were either trained on the 2007 train/val set (07) or 2007+2012 train/val sets (07+12) and evaluated on the 2007 test set.

| Official Networks | mAP (07) | mAP (07+12) | FPS |
|:-|:-:|:-:|:-:|
| SSD300 | 68.0 | 74.3 | N/A |
| SSD300 DCT | 39.2 | 47.8 | N/A |

| Networks, VGG based (our trainings) | mAP (07) | mAP (07+12) | FPS |
|:-|:-:|:-:|:-:|
| SSD300 | 65.0 | 74.0 | 102 |
| SSD300 DCT | 48.9 | 60.0 | 262 |
| SSD300 DCT Y | 50.7 | 59.8 | 278 |
| SSD300 DCT Deconvolution | 38.4 | 53.5 | 282 |

| Network, ResNet50 based (our trainings) | mAP (07) | mAP (07+12) | FPS |
|:-|:-:|:-:|:-:|
| SSD300-Resnet50 (retrained) | 61.3 | 73.1 | 108 |
| SSD300 DCT LC-RFA | 61.7 | 70.7 | 110 |
| SSD300 DCT LC-RFA Y | 62.1 | 71.0 | 109 |
| SSD300 DCT LC-RFA-Thinner | 58.5 | 67.5 | 176 |
| SSD300 DCT LC-RFA-Thinner Y | 60.6 | 70.2 | 174 |
| SSD300 DCT Deconvolution-RFA | 54.7 | 68.8 | 104 |

### Training on the PASCAL VOC dataset

The data can be downloaded on the [official](http://host.robots.ox.ac.uk:8080/pascal/VOC/) website.

After downloading you should have directories following this architecture:

```text
VOCdevkit
|
|_ VOC2007
|  |_ Annotations
|  |_ ImageSets
|  |_ JPEGImages
|  |_ ...
|
|_ VOC2012
   |_ Annotations
   |_ ImageSets
   |_ JPEGImages
   |_ ...
```

Then you'll just need to set the configuration files to fit your needs and follow the procedure described in the [training](##Training) section. **The hyper-parameters provided for the training were not used in a parallel setting.**

Also the system variable should be set to the Pascal VOC folder (if you use the provided config files)

```bash
# Setting the main dirs for the training datasets
export DATASET_PATH=<path_to_directory>/VOCdevkit
```

## Detection (MS-COCO)

### Results on the MS-COCO dataset

Results for training on the Pascal VOC dataset are presented bellow. Networks were either trained on the 2007 train/val set (07) or 2007+2012 train/val sets (07+12) and evaluated on the 2007 test set.

| Official Networks | mAP 0.5:0.95 |
|:-|:-:|
| SSD300 | 23.2 |

| Networks, VGG based (our trainings) | mAP  |
|:-|:-:|
| SSD300 | 24.5 |
| SSD300 DCT | 14.3 |
| SSD300 DCT Y | 14.4 |
| SSD300 DCT Deconvolution | 13.5 |

| Network, ResNet50 based (our trainings) | mAP (07) |
|:-|:-:|
| SSD300-Resnet50 (retrained) | 26.8 |
| SSD300 DCT LC-RFA | 25.8 |
| SSD300 DCT LC-RFA Y | 25.2 |
| SSD300 DCT LC-RFA-Thinner | 25.4 |
| SSD300 DCT LC-RFA-Thinner Y | 24.6 |
| SSD300 DCT Deconvolution-RFA | 25.9 |

### Training on the MS-COCO dataset

The data can be downloaded on the [official](https://cocodataset.org/#home) website.

After downloading you should have directories following this architecture:

```text
mscoco
|
|_ annotations
|  |_ captions_train2014.json
|  |_ instances_train2017.json
|  |_ person_keypoints_val2017.json
|  |_ ...
|
|_ train2017
|  |_ 000000110514.jpg
|  |_ ...
|
|_ val2017
|  |_ ...
|
|_ test2017
   |_ ...
```

Then you'll just need to set the configuration files to fit your needs and follow the procedure described in the [training](##Training) section. **The hyper-parameters provided for the training were not used in a parallel setting.**

Also the system variable should be set to the mscoco folder (if you use the provided config files)

```bash
# Setting the main dirs for the training datasets
export DATASET_PATH=<path_to_directory>/VOCdevkit
```


## Running the documentation for a deeper usage of the provided code

I know from experience that diving into ones code to adapt to its own project is often hard and confusing at first. To help you if you ever want to toy with the code, a built-in documentation is provided. It uses a modify version of the keras documentation generator ([here](https://github.com/D3lt4lph4/pythondoc)).

To generate the documentation:

```bash
pip install mkdocs

cd docs

python autogen.py
```

To display the documentation:

```bash
# From root of the repository
mkdocs serve
```

## Method limitations

The presented method has some limitations especially for general purpose deployments. The two main issues I see are described hereafter.

### Image Resizing

Resizing images in the RGB domain is straightforward whereas resizing in the DCT domain is more complicated. Although theoretically doable, methods for such usage are not implemented. The following list of articles explore the possibility to resize images directly in the frequency domain:

- [On Resizing Images In The DCT Domain](https://ieeexplore.ieee.org/document/1421685)
- [Image Resizing In The Discrete Cosine Transform Domain](https://ieeexplore.ieee.org/document/537460)
- [Fast Image Resizing in Discrete Cosine Transform Domain with Spatial Relationship between DCT Block and its Sub-Blocks](https://ieeexplore.ieee.org/document/4590237)
- [Design and Analysis of an Image Resizing Filter in the Block-DCT Domain](https://www.researchgate.net/publication/3308607_Design_and_Analysis_of_an_Image_Resizing_Filter_in_the_Block-DCT_Domain)

For classification, the impact is limited as long as the images are about the same size as the original training images. This is due to the fact that the network can be made fully convolutionnals. For detection, this is a bit more complicated as the SSD in the presented implementation does not scale well (although it should theoretically be able to do so). This is due to the original design of the network and the need for padding layers. I intend to test modified version of the network if I find some time to do so.
 
### Training Pipeline

The second limitation is for training. Data-augmentation has to be carried in the RGB domain, thus the data-augmentation pipeline is the following one: JPEG => RGB => data-augmentation => JPEG => Compressed Input. This slows the training down.

## TO DO

- Add displayers to the config files for the PASCAL VOC and MS COCO datasets
- Add displayer with gt for the PASCAL VOC and MS COCO datasets
- Use correct path for the PASCAL VOC and MS COCO datasets
- Set correct descriptions for all the config files
