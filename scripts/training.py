import sys
from os import mkdir, listdir, environ, makedirs
from os.path import join, dirname, isfile, expanduser, basename, normpath
from shutil import copyfile
import argparse
import string
import random
import csv

from operator import itemgetter

import keras.backend as K
from keras.models import load_model

from jpeg_deep.layers.ssd_layers import AnchorBoxes, DecodeDetections, L2Normalization

import tensorflow as tf
try:
    import horovod.keras as hvd
except ImportError:
    print("Failed to import Horovod, skipping the module.")

parser = argparse.ArgumentParser(
    description="Start or restart a training given a configuration file.")
parser.add_argument(
    '-r', '--restart', help="Experiment to restart. The experiment checkpoint folder must contain only the last weights.")
parser.add_argument('-c', '--configuration',
                    help="Path to the directory containing the config file to use. The configuration file should be named 'config_file.py' (see the examples in the config folder of the repository).")

feature_parser = parser.add_mutually_exclusive_group(required=True)
feature_parser.add_argument('--horovod', dest='horovod', action='store_true')
feature_parser.add_argument(
    '--no-horovod', dest='horovod', action='store_false')
parser.set_defaults(feature=True)

args = parser.parse_args()


if args.horovod:
    hvd.init()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config_tf.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config_tf))
    verbose = 1 if hvd.rank() == 0 else 0
else:
    verbose = 1

if args.restart:
    sys.path.append(join(args.restart, "config"))
    from saved_config import TrainingConfiguration
    config = TrainingConfiguration()

    # In case of restart save in the experiment directory.
    output_dir = basename(normpath(args.restart))
else:
    # Load configuration file
    sys.path.append(args.configuration)
    from config_file import TrainingConfiguration
    config = TrainingConfiguration()

    # Generate random key for experiment.
    key = ''.join(random.choice(string.ascii_uppercase +
                                string.ascii_lowercase + string.digits) for _ in range(32))

    output_dir = "{}_{}_{}".format(config.workspace, config.project_name, key)

if (args.horovod and hvd.rank() == 0) or (not args.horovod):
    output_dir = join(environ["EXPERIMENTS_OUTPUT_DIRECTORY"], output_dir)

    checkpoints_output_dir = join(output_dir, "checkpoints")
    config_output_dir = join(output_dir, "config")
    logs_output_dir = join(output_dir, "logs")

    # We create all the output directories
    makedirs(output_dir, exist_ok=True)
    makedirs(checkpoints_output_dir, exist_ok=True)
    makedirs(config_output_dir, exist_ok=True)
    makedirs(logs_output_dir, exist_ok=True)

    directories_dict = {"output": output_dir, "checkpoints_dir": checkpoints_output_dir,
                        "config_dir": config_output_dir, "log_dir": logs_output_dir}

# Prepare the generators
config.prepare_training_generators()

# Loading the model
model = config.network

if args.horovod:
    config.prepare_horovod(hvd)

if (args.horovod and hvd.rank() == 0) or (not args.horovod):
    config.prepare_runtime_checkpoints(directories_dict)

    # Saving the config files, the temps file is here to modification on the fly if required after (for evaluation for instance).
    if args.restart:
        copyfile(join(args.restart, "config", "saved_config.py"),
                 join(config_output_dir, "saved_config.py"))
        copyfile(join(args.restart, "config", "saved_config.py"),
                 join(config_output_dir, "temp_config.py"))
    else:
        copyfile(join(args.configuration, "config_file.py"),
                 join(config_output_dir, "saved_config.py"))
        copyfile(join(args.configuration, "config_file.py"),
                 join(config_output_dir, "temp_config.py"))

# If weights and first worker in case of hvd, if weights else.
if (config.weights is not None and args.horovod and hvd.rank() == 0) or config.weights is not None and not args.horovod:
    # Loading the weights
    if args.restart:
        # In case of restart, get the weights
        model_file = [f for f in listdir(
            join(args.restart, "checkpoints")) if isfile(join(args.restart, "checkpoints", f))][0]
        model_path = join(args.restart, "checkpoints", model_file)
        restart_epoch = int(model_file.split("_")[0].split('-')[-1])
        print("Loading weights (by name): {}".format(model_path))
        K.clear_session()

        # TODO: Move the custom objects to the conf file.
        model = load_model(model_path, custom_objects={
                           "L2Normalization": L2Normalization, "DecodeDetections": DecodeDetections, "AnchorBoxes": AnchorBoxes, "compute_loss": config.loss})
    else:
        print("Loading weights (by name): {}".format(config.weights))
        model.load_weights(config.weights, by_name=True)

if args.restart:
    # Fit the model on the batches generated.
    model.fit_generator(config.train_generator,
                        validation_data=config.validation_generator,
                        epochs=config.epochs,
                        steps_per_epoch=config.steps_per_epoch,
                        callbacks=config.callbacks,
                        workers=config.workers,
                        verbose=verbose,
                        initial_epoch=restart_epoch,
                        validation_steps=config.validation_steps,
                        use_multiprocessing=config.multiprocessing)
else:
    # Compiling the model
    model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics)

    # Fit the model on the batches generated.
    model.fit_generator(config.train_generator,
                        validation_data=config.validation_generator,
                        epochs=config.epochs,
                        steps_per_epoch=config.steps_per_epoch,
                        callbacks=config.callbacks,
                        workers=config.workers,
                        verbose=verbose,
                        validation_steps=config.validation_steps,
                        use_multiprocessing=config.multiprocessing)
