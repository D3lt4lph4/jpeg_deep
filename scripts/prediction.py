from os import getcwd
from os.path import join

import argparse
import sys
sys.path.append(getcwd())

parser = argparse.ArgumentParser("Runs a prediction pass on a trained network.")
parser.add_argument("experiment", help="The experiment directory.")
parser.add_argument("weights", help="The weights to load.")
parser.add_argument('-gt', '--groundTruth', action='store_true', help="If the ground truth should be displayed.")
args = parser.parse_args()

# We reload the config to get everything as in the experiment
sys.path.append(join(args.experiment, "config"))
from temp_config import TrainingConfiguration
config = TrainingConfiguration()

# Loading the model, for prepare for inference if required
config.prepare_for_inference()
config.network.load_weights(args.weights)

model = config.network

model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=config.metrics)

config.prepare_testing_generator()
config.test_generator.shuffle = True

# Getting the batch to process
X, _ = config.test_generator.__getitem__(0)

# If the input is not a displayable stuff, get the displayable
X_true, y = config.test_generator.get_raw_input_label(0)

y_pred = model.predict(X)

if args.groundTruth:
    config.displayer.display_with_gt(y_pred, X_true, y)
else:
    config.displayer.display(y_pred, X_true)
