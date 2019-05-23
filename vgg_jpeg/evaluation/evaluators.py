#from template.evaluators import TemplateEvaluator

import numpy as np

class Evaluator(object):

    def __init__(self, generator=None):
        self.score = None
        self._generator = generator
        self.runs = False
        self.number_of_runs = None

    def __call__(self, model, test_generator=None):
        if self._generator is None and test_generator is None:
            raise RuntimeError("A generator should be specified using the init or parameters.")
        self.runs = False
        if test_generator is not None:
            self._generator = test_generator

        self.score = model.evaluate_generator(self._generator, verbose=1)

    def make_runs(self, model, test_generator=None, number_of_runs=10):

        if self._generator is None and test_generator is None:
            raise RuntimeError("A generator should be specified using the init or parameters.")

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
            return "Number of runs: {}\nAverage score: {}".format(self.number_of_runs, self.score)
        else:
            return "The evaluated score is {}.".format(self.score)

    @property
    def test_generator(self):
        return self._generator
    
    def display_results(self):
        pass
