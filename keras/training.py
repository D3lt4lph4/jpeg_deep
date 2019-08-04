""" 
The script can:

    - log the experiment on comet.ml
    - create a config file locally with the configuration in it
    - create a csv file with the val_loss and train_loss locally
    - save the checkpoints locally
"""
import sys
from os import mkdir, listdir, environ, makedirs
from os.path import join, dirname, isfile, expanduser
from shutil import copyfile
import argparse
import string
import random
import csv

from operator import itemgetter

from comet_ml import Experiment

import keras.backend as K

import tensorflow as tf
import horovod.keras as hvd

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--restart', help="Restart the training from a previous stopped config. The argument is the path to the experiment folder.", type=str)
parser.add_argument('-c', '--configuration', help="Path to the directory containing the config file to use. The configuration file should be named 'config_file.py' (see the examples in the config folder of the repository).")
parser.add_argument('--comet', dest='comet', action='store_true')
parser.add_argument('--no-comet', dest='comet', action='store_false')
parser.add_argument('--myria', dest='myria', action='store_true')
parser.add_argument('--no-myria', dest='myria', action='store_false')
parser.set_defaults(feature=False)
parser.add_argument('-ji', '--jobid', type=int, help="When submitting through sbatch, a job id might be specified. If so, the id will be stored in a job.txt file in the LOG_DIRECTORY")
parser.add_argument('--horovod', dest='horovod', action='store_true')
args = parser.parse_args()

if args.horovod:
    hvd.init()

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config_tf.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config_tf))

# Keras variable if restart
restart_epoch = None
restart_lr = None
key = ""

# If we restart an experiment, no need to check for a configuration, we load the one from the config file.
if args.restart is not None:
    sys.path.append(join(args.restart, "config"))
    from saved_config import TrainingConfiguration
    config = TrainingConfiguration()
    key = dirname(args.restart).split("_")[-1]

    # The results will be saved in an other directory not to pollute the original one.
    key += "_temp"

    # We extract the last saved weight and the corresponding epoch
    weights_path = join(args.restart, "checkpoints")
    weights_files = sorted([[f, int(f.split('_')[0].split('-')[1])] for f in listdir(weights_path) if isfile(join(weights_path, f))], key=itemgetter(1))


    config.weights = join(weights_path, weights_files[-1][0])
    restart_epoch = weights_files[-1][1]
    
    # We load the results file to get the epoch learning rate.
    with open(join(args.restart, 'results/results.csv'), newline='') as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        data = []
        for row in results:
            data.append(row)
        lr_index = data[0].index('lr')
        restart_lr = float(data[restart_epoch][lr_index])
        print(data[restart_epoch])

    output_dir = "{}_{}_{}".format(config.workspace, config.project_name, key)

else:
    sys.path.append(args.configuration)
    from config_file import TrainingConfiguration
    config = TrainingConfiguration()

    # Starting the experiment
    if args.comet:
        experiment = Experiment(api_key=environ["COMET_API_KEY"],
                                project_name=config.project_name, workspace=config.workspace)
        key = experiment.get_key()

    else:
        key = ''.join(random.choice(string.ascii_uppercase +
                                        string.ascii_lowercase + string.digits) for _ in range(32))

    output_dir = "{}_{}_{}".format(config.workspace, config.project_name, key)


output_dir = join(environ["EXPERIMENTS_OUTPUT_DIRECTORY"], "{}_{}_{}".format(config.workspace, config.project_name, key))

checkpoints_output_dir = join(output_dir, "checkpoints")
config_output_dir = join(output_dir, "config")
results_output_dir = join(output_dir, "results")

# We create all the output directories
makedirs(output_dir, exist_ok=True)
makedirs(checkpoints_output_dir, exist_ok=True)
makedirs(config_output_dir, exist_ok=True)
makedirs(results_output_dir, exist_ok=True)
makedirs(environ["LOG_DIRECTORY"], exist_ok=True)

if args.jobid is not None:
    with open(join(environ["LOG_DIRECTORY"], "job.txt"), "a+") as text_file:
        print("{} => {}".format(key, args.jobid), file=text_file)

if args.horovod:
    config.prepare_horovod(hvd)

config.add_csv_logger(results_output_dir)
config.add_model_checkpoint(checkpoints_output_dir)

# Saving the config file.
if args.restart is None:
    copyfile(join(args.configuration, "config_file.py"), join(config_output_dir, "saved_config.py"))
    copyfile(join(args.configuration, "config_file.py"), join(config_output_dir, "temp_config.py"))
else:
    copyfile(join(args.restart, "config/saved_config.py"), join(config_output_dir, "saved_config.py"))
    copyfile(join(args.restart, "config/saved_config.py"), join(config_output_dir, "temp_config.py"))

# Logging the experiment
if args.restart is None:
    if args.comet:
        experiment.log_parameters(config.__dict__)

# Creating the model
model = config.network

if config.weights is not None:
    print("Loading weights (by name): {}".format(config.weights))
    model.load_weights(config.weights, by_name=True)

# Setting the iteration variable if restarting the training.
if args.restart is not None:
    config.optimizer.iterations = K.variable(config.steps_per_epoch * restart_epoch, dtype='int64', name='iterations')
    config.optimizer.lr = K.variable(restart_lr, name='lr')

# Prepare the generators
config.prepare_training_generators()

# Compiling the model
model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics)

if restart_epoch is not None:
    model.fit_generator(config.train_generator,
                        validation_data=config.validation_generator,
                        epochs=config.epochs,
                        steps_per_epoch=config.steps_per_epoch,
                        callbacks=config.callbacks,
                        workers=config.workers,
                        use_multiprocessing=config.multiprocessing,
                        initial_epoch=restart_epoch)
else:
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(config.train_generator,
                        validation_data=config.validation_generator,
                        epochs=config.epochs,
                        steps_per_epoch=config.steps_per_epoch,
                        callbacks=config.callbacks,
                        workers=config.workers,
                        validation_steps=config.validation_steps,
                        use_multiprocessing=config.multiprocessing)
