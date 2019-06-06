""" 
The script will:

    - log the experiment on comet.ml
    - create a config file locally with the configuration in it
    - create a csv file with the val_loss and train_loss locally
    - save the checkpoints locally
"""
import sys
from os import mkdir, listdir, environ
from os.path import join, dirname, isfile, expanduser
from shutil import copyfile
import argparse
import string
import random
import csv

from operator import itemgetter

from comet_ml import Experiment

import keras
from keras.utils import multi_gpu_model
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--restart', help="Restart the training from a previous stopped config. The argument is the path to the experiment folder.", type=str)
parser.add_argument('-c', '--configuration', help="Path to the config file to use.")
parser.add_argument('--comet', dest='comet', action='store_true')
parser.add_argument('--no-comet', dest='comet', action='store_false')
parser.add_argument('--myria', dest='myria', action='store_true')
parser.add_argument('--no-myria', dest='myria', action='store_false')
parser.set_defaults(feature=False)
parser.add_argument('-ji', '--jobid', type=int)
args = parser.parse_args()

# Class to be able to train with multiple gpus and use the checkpoints
class ModelMultiGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMultiGPU, self).__getattribute__(attrname)

# Keras variable if restart
restart_epoch = None
restart_lr = None
key = ""

output_dir = args.restart

# If we restart an experiment, no need to check for a configuration, we load the one from the config file.
if output_dir is not None:
    sys.path.append(join(args.restart, "config"))
    from saved_config import TrainingConfiguration
    config = TrainingConfiguration()
    key = dirname(args.restart).split("_")[-1]


    # We load the results file to get the epoch learning rate.
    with open(join(output_dir, 'results/results.csv'), newline='') as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        data = []
        for row in results:
            data.append(row)
        lr_index = data[0].index('lr')
        restart_lr = float(data[-1][lr_index])
        restart_epoch = len(data) - 1

    # We extract the last saved weight and the corresponding epoch
    weights_path = join(output_dir, "checkpoints")
    weights_files = sorted([[f, int(f.split('_')[0].split('-')[1])] for f in listdir(weights_path) if isfile(join(weights_path, f))], key=itemgetter(1))


    config.weights = join(weights_path, weights_files[-1][0])

else:
    sys.path.append(args.configuration)
    from config_file import TrainingConfiguration
    config = TrainingConfiguration()

    # Starting the experiment
    if args.comet:
        experiment = Experiment(api_key="F5aa0Le4aKpPPyCiGBIUrvfQ0",
                                project_name=config.project_name, workspace=config.workspace)
        key = experiment.get_key()

    else:
        key = ''.join(random.choice(string.ascii_uppercase +
                                        string.ascii_lowercase + string.digits) for _ in range(32))

    output_dir = "{}_{}_{}".format(config.workspace, config.project_name, key)

if args.myria:
    output_dir = join(environ["LOCAL_WORK_DIR"], "{}_{}_{}".format(config.workspace, config.project_name, key))
checkpoints_output_dir = join(output_dir, "checkpoints")
config_output_dir = join(output_dir, "config")
results_output_dir = join(output_dir, "results")

# We create all the output directories
if args.restart is None or args.myria:
    mkdir(output_dir)
    mkdir(checkpoints_output_dir)
    mkdir(config_output_dir)
    mkdir(results_output_dir)

if args.jobid is not None:
    with open(join(expanduser("~"), "job.txt"), "a+") as text_file:
        print("{} => {}".format(key, args.jobid), file=text_file)

config.add_csv_logger(results_output_dir)
config.add_model_checkpoint(checkpoints_output_dir)

# Saving the config file.
if args.restart is None:
    copyfile(join(args.configuration, "config_file.py"), join(config_output_dir, "saved_config.py"))
    copyfile(join(args.configuration, "config_file.py"), join(config_output_dir, "temp_config.py"))

if args.restart is not None and args.myria:
    copyfile(join(args.restart, "config/saved_config.py"), join(config_output_dir, "saved_config.py"))
    copyfile(join(args.restart, "config/saved_config.py"), join(config_output_dir, "temp_config.py"))

# Logging the experiment
if args.restart is None:
    if args.comet:
        experiment.log_parameters(config.__dict__)

# Creating the model
model = config.network

if config.weights is not None:
    print("Reloading weights: {}".format(config.weights))
    model.load_weights(config.weights, by_name=True)

if config.gpus == 0 or config.gpus == 1:
    model_gpu = model
else:
    model_gpu = model #ModelMultiGPU(model, config.gpus)

# Setting the iteration variable if restarting the training.
if args.restart is not None:
    config.optimizer.iterations = K.variable(config.steps_per_epoch * restart_epoch, dtype='int64', name='iterations')
    config.optimizer.lr = K.variable(restart_lr, name='lr')

# Prepare the generators
config.prepare_training_generators()

config.optimizer = keras.optimizers.Adadelta(lr=1.0 * hvd.size())
config.optimizer = hvd.DistributedOptimizer(config.optimizer)

# Compiling the model
model_gpu.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics)
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('/home/2017018/bdegue01/checkpoint-{epoch}.h5'))

if restart_epoch is not None:
    model_gpu.fit_generator(config.train_generator,
                            validation_data=config.validation_generator,
                            epochs=config.epochs,
                            steps_per_epoch=config.steps_per_epoch // hvd.size(),
                            callbacks=config.callbacks,
                            workers=config.workers,
                            use_multiprocessing=config.multiprocessing,
                            initial_epoch=restart_epoch)
else:
    # Fit the model on the batches generated by datagen.flow().
    model_gpu.fit_generator(config.train_generator,
                            validation_data=config.validation_generator,
                            epochs=config.epochs,
                            steps_per_epoch=config.steps_per_epoch // hvd.size(),
                            callbacks=config.callbacks,
                            workers=config.workers,
                            use_multiprocessing=config.multiprocessing,
                            validation_steps=1000 // hvd.size())

model.save(join(checkpoints_output_dir, "final.h5"))
