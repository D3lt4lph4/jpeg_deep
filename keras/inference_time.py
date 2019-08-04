import argparse

from os.path import join

import sys

from config.dummy.config_file import TrainingConfiguration as TrainingConfigurationDummy

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='cmd')
parser.add_argument("-nr", "--numberOfRun", help="The number of time the generator should be run, the results will be average on this number.", type=int, default=10)

parser_experiment = subparser.add_parser('experiment')
parser_experiment.add_argument("experiment", help="The experiment directory.")
parser_experiment.add_argument("weights", help="The weights to load.")

parser_experiment = subparser.add_parser('dummy')

args = parser.parse_args()

if args.cmd == "dummy":
    config = TrainingConfigurationDummy()
else:
    sys.path.append(join(args.experiment, "config"))
    from temp_config import TrainingConfiguration
    config = TrainingConfiguration()

config.prepare_for_inference()
config.prepare_evaluator()
config.prepare_testing_generator()

# Loading the model
model = config.network
if args.cmd == "experiment":
    model.load_weights(args.weights)

model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=config.metrics)

print("We evaluate a model with {} parameters.".format(model.count_params()))
evaluator = config.evaluator

evaluator.make_runs(model, config.test_generator, args.numberOfRun)

print(evaluator)
