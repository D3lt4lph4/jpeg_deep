from template_keras.evaluators import TemplateEvaluator

import numpy as np
import time
from statistics import mean, stdev

from tqdm import tqdm


class Evaluator(TemplateEvaluator):
    def __init__(self, generator=None):
        self.score = None
        self._generator = generator
        self.runs = False
        self.number_of_runs = None

    def __call__(self, model, test_generator=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        self.score = model.evaluate_generator(self._generator, verbose=1)

    def model_speed(self, model, test_generator=None, number_of_runs=10, iteration_per_run=100):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )
        if test_generator is not None:
            self._generator = test_generator

        times = []

        X, _ = self._generator.__getitem__(0)

        for _ in tqdm(range(number_of_runs)):
            start_time = time.time()
            for _ in range(iteration_per_run):
                _ = model.predict(X)
            times.append(time.time() - start_time)

        print("It took {} seconds on average of {} runs to run {} iteration of prediction with bacth size {}.".format(
            mean(times), number_of_runs, iteration_per_run, self._generator.batch_size))
        print("The number of FPS for the tested network was {}.".format(
            self._generator.batch_size * iteration_per_run / mean(times)))

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError(
                "A generator should be specified using the init or parameters."
            )

        scores = []
        self.runs = True

        if test_generator is not None:
            self._generator = test_generator

        if test_generator is not None:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))
        else:
            for i in range(number_of_runs):
                scores.append(model.evaluate_generator(self._generator))

        self.score = np.mean(np.array(scores), axis=0)
        self.number_of_runs = number_of_runs

    def __str__(self):
        if self.runs:
            return "Number of runs: {}\nAverage score: {}".format(
                self.number_of_runs, self.score)
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator

    def display_results(self):
        print("The evaluated score is {}.".format(self.score))
