# VGG for JPEG compressed files

This project aims to to train a classification neural network using compressed JPEG files.

## How to install

```bash
cd deep_template/template_keras/template_keras

python setup.py install

cd ../../..

pip install requirements.txt
```

## How to start the calculations

```bash
# Use the pre-set config file (training neural network in RGB)
python training.py -c config/vggA/

# Use the provided slurm script (modify to your needs)
sbatch vgg_jpeg.sl
```