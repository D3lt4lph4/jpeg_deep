# Neural Networks using compressed JPEG images.

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

python setup.py install

# Back to main directory
cd ../../..

# Will install all the dependencies, you may want to change tensorflow version depending what's installed on your computer.
pip install -r requirements.txt --user
```

## Slurm

Most of the calculation were run on multiple GPU on a supercomputer, the repository provides an example script to train the networks.



## Classification

## How to start the calculations: Slurm

```bash
# Use the pre-set config file (training neural network in RGB)
python training.py -c config/vggA/

# Use the provided slurm script (modify to your needs)
sbatch vgg_jpeg.sl
```

### Detection

