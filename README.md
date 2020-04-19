# Neural Networks using compressed JPEG images.

This repository provides code to train and used neural network on compressed JPEG images. **No pre-trained weights are/will be made available.**

This implementation relies on the module [jpeg2dct](https://github.com/uber-research/jpeg2dct) from uber research team. The SSD used in this repository was taken from [this repository](https://github.com/pierluigiferrari/ssd_keras) and then modified.

The following networks (and modified versions) are available for usage in the repository:

- [VGG16](https://arxiv.org/abs/1409.1556)
- [ResNet50](https://arxiv.org/abs/1512.03385)
- [SSD](https://arxiv.org/abs/1512.02325)


## Compressed representation and limitations

### Image Resizing

Using the compressed representation of the data brings some limitations. The main limitation discussed here is the resizing of the input data. Resizing images in the RGB domain is straightforward whereas resizing in the DCT domain is more complicated. The following list of articles explore the possibility to resize images directly in the frequency domain:

- [On Resizing Images In The DCT Domain](https://ieeexplore.ieee.org/document/1421685)
- [Image Resizing In The Discrete Cosine Transform Domain](https://ieeexplore.ieee.org/document/537460)
- [Fast Image Resizing in Discrete Cosine Transform Domain with Spatial Relationship between DCT Block and its Sub-Blocks](https://ieeexplore.ieee.org/document/4590237)
- [Design and Analysis of an Image Resizing Filter in the Block-DCT Domain](https://www.researchgate.net/publication/3308607_Design_and_Analysis_of_an_Image_Resizing_Filter_in_the_Block-DCT_Domain)

These methods are so far not implemented in this repository.
 
### Training Pipeline

The second limitation is for training. Data-augmentation has to be carried in the RGB domain, thus the data-augmentation pipeline is the following one: JPEG => RGB => data-augmentation => JPEG => Compressed Input. This slows down the training.

## Installation

The installation steps are the same for classification and detection:

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

# If you intend to display stuff
pip install PyQt5
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
```

### Starting the training

Once you have defined all the variables and modified the config files to your needs, simply run the following command (you will need to update some of the parameters to when not using horovod):

```bash
python scripts/training.py -c <config_dir_path> --no-horovod
```

The config file in the <config_dir_path> needs to be named "config.py" for the script to run correctly.

For more details on classification training on ImageNet dataset, refer to this [section](##classification), for more details for training on Pascal VOC dataset, refer to this [section](##Detection-Pascal-VOC) and for more details for training MS-COCO dataset, refer to this [section](##Detection-MS-COCO)

### Training using horovod

The training script support the usage of horovod. An exemple file for training with horovod using slurm is provided [jpeg_deep.sl](slurm/jpeg_deep.sl).

```bash
cd slurm
sbatch jpeg_deep.sl
```

If you do not run on a multi-cluster computation facility that uses slurm, please refer to the original [horovod git](https://github.com/horovod/horovod)

## Predictions

**No pre-trained weights are/will be made available.** To get this section running, you'll have to retrain the networks from scratch.

### Display the results

Displaying the results can be done using the [prediction.py](scripts/prediction.py) script. In order to use the script you have to first carry a training for at least one epoch (the prediction pre-suppose that you have an experiment folder).

The prediction will be done on the test set. You need to modify the config_temp.py file in the experiment generated folder in order to use a different dataset.

**For the vgg16 based classifiers:** The prediction script uses the test generator specified in the config file to get the data. Hence, with the provided examples, you may need first to convert the weights to a fully convolutional network.

Once this is done, simply run the following command:

```bash
python scripts/prediction.py <experiment_path> <weights_path>
```

### Prediction time

To do

## Classification on ImageNet

### Results

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
| VGG-DCT | 62.6 | 84.6 | 583 |
| VGG-DCT | 65.9 | 86.7 | 571 |

| ResNet50 based Newtorks (our trainings) | top-1 | top-5 | FPS |
|:-|:-:|:-:|:-:|
| ResNet50 | 74.73 | 92.33 | 324 |
| LC-RFA | **74.82** | **92.58** | 318 |
| LC-RFA Y | 73.25 | 91.40 | 329 |
| LC-RFA-Thinner | 74.62 | 92.33 | 389 |
| LC-RFA-Thinner Y | 72.48 | 91.04 | 395 |
| Deconvolution-RFA | 74.55 | 92.39 | 313 |

### Training on ImageNet

The dataset can be downloaded [here](https://arxiv.org/abs/1904.08408). Choose the version that suits your needs, I used the 2012 (Object Detection) data.

Once the data is downloaded, to use the provided generators, it should be stored following this tree (as long as you have separeted train and validation folders you should be okay)

```text

```

Then you'll just need to set the configuration files to fit your needs and follow the procedure described in the [training](##Training) section. Keep in mind that the provided configuration files were used in a distributed training, hence the hyper parameters fit this particular settings. If you don't train that way, you'll want to change them.




## Detection Pascal VOC

### Details in the dataset path


## Detection MS-COCO

### Details in the dataset path

## Running the documentation for a deeper usage of the provided code
