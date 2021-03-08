
import argparse

from os import getcwd

import sys
sys.path.append(getcwd())
from os.path import join

parser = argparse.ArgumentParser(
    "Evaluate the result of a training given the parameters in the temp_config.py file.")
parser.add_argument("experiment", help="The experiment directory.")
parser.add_argument(
    "weights", help="The weights to be loaded into the network, should be in the checkpoint folder of the experiment.")
parser.add_argument("-s", "--submission",
                    help="If specified, tells that the evaluator should run the prediction and write files in submission format.", action="store_true")
parser.add_argument(
    '-o', "--output", help="Is used only when making predictions for a submission. Where to write the output folder result.", default=".")
parser.add_argument("-st", "--steps", help="The number of files to process for the evaluation (mainly for debug).", type=int, default=None)
args = parser.parse_args()

sys.path.append(join(args.experiment, "config"))
from temp_config import TrainingConfiguration
config = TrainingConfiguration()

config.prepare_for_inference()
config.prepare_testing_generator()
config.prepare_evaluator()

model = config.network

model.load_weights(args.weights)

model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=config.metrics)

evaluator = config.evaluator

if not args.submission:
    evaluator(model, config.test_generator, args.steps)
    # If anything to display, apart from the results (graphs,...)
    evaluator.display_results()
else:
    evaluator.predict_for_submission(
        model, config.test_generator, args.output)
