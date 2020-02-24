import argparse

from os.path import join

import sys

parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="The experiment directory.")
parser.add_argument("-nr", "--numberOfRun",
                    help="The number of time the generator should be run, the results will be average on this number.", type=int, default=10)
parser.add_argument("-w", "--weights", help="The weights to load.", default=None)


args = parser.parse_args()

sys.path.append(join(args.experiment, "config"))
try:
    from temp_config import TrainingConfiguration
    config = TrainingConfiguration()
except ModuleNotFoundError:
    sys.path.append(args.experiment)
    from config_file import TrainingConfiguration
    config = TrainingConfiguration()

config.prepare_for_inference()
config.prepare_evaluator()
config.prepare_testing_generator()

# Loading the model
model = config.network
if args.weights is not None:
    model.load_weights(args.weights)

model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=config.metrics)

print("We evaluate a model with {} parameters.".format(model.count_params()))

evaluator = config.evaluator

evaluator.model_speed(model, config.test_generator, args.numberOfRun)
