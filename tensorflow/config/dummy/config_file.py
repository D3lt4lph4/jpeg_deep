from os.path import join

from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist

from template.networks import MNISTExample
from template.generators import DummyGenerator
from template.displayer import MNISTDisplayer
from template.evaluators import Evaluator

from template.config import TemplateConfiguration

class TrainingConfiguration(TemplateConfiguration):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = "This is the template config file."
        self.experiment_description = "Dummy experiment. Only here to test the template file. It trains the keras [exemple](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) on the MNIST dataset"

        # System dependent variable
        self._workers = 1
        self._multiprocessing = False
        self._gpus = 1

        # Variables for comet.ml
        self._project_name = "general"
        self._workspace = "d3lt4lph4"
        self.output_dir = join(environ["EXPERIMENTS_OUTPUT_DIRECTORY"], "{}_{}_{}".format(config.workspace, config.project_name, key))
        # Network variables
        self.num_classes = 10
        self.img_size = (28, 28)
        self._weights = None
        self._network = MNISTExample(self.num_classes)

        # Training variables
        self._epochs = 12
        self._batch_size = 128
        self._steps_per_epoch = 60000 // 128
        self._optimizer = Adadelta()
        self._loss = categorical_crossentropy
        self._metrics = ['accuracy']

        # Keras stuff
        self.model_checkpoint = None
        self.csv_logger = None
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping_params = {"monitor":'val_loss', "min_delta":0, "patience":7}
        self.early_stopping = EarlyStopping(**self.early_stopping_params)
        self.reduce_lr_on_plateau_params = {"monitor":'val_loss', "factor":0.1, "patience":5}
        self.reduce_lr_on_plateau = ReduceLROnPlateau(**self.reduce_lr_on_plateau_params)

        self._callbacks = [self.terminate_on_nan, self.early_stopping, self.reduce_lr_on_plateau]

        # Creating the training and validation generator
        self.num_batches = 1000

        self.train_generator_params = {"num_batches":self.num_batches, "batch_size":self.batch_size}
        self.validation_generator_params = {"num_batches":self.num_batches, "batch_size":self.batch_size}
        self.test_generator_params = {"num_batches":self.num_batches, "batch_size":self.batch_size}

        self._train_generator = None
        self._validation_generator = None
        self._test_generator = None
        self._evaluator = None
        self._displayer = MNISTDisplayer()

    def add_csv_logger(self, output_path, filename="results.csv", separator=',', append=True):
        self.csv_logger = CSVLogger(filename=join(output_path, filename), separator=separator, append=append)
        self.callbacks.append(self.csv_logger)

    def add_model_checkpoint(self, output_path, verbose=1, save_best_only=True):

        self.model_checkpoint = ModelCheckpoint(filepath=join(output_path, "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"), verbose=verbose, save_best_only=save_best_only)
        self.callbacks.append(self.model_checkpoint)

    def prepare_for_inference(self):
        pass
    
    def prepare_evaluator(self):
        self._evaluator = Evaluator()
    
    def prepare_testing_generator(self):
        self._test_generator = DummyGenerator(**self.test_generator_params)

    def prepare_training_generators(self):
        self._train_generator = DummyGenerator(**self.train_generator_params)
        self._validation_generator = DummyGenerator(**self.validation_generator_params)

    @property
    def train_generator(self):
        return self._train_generator

    @train_generator.setter
    def train_generator(self, value):
        self._train_generator = value

    @property
    def validation_generator(self):
        return self._validation_generator

    @validation_generator.setter
    def validation_generator(self, value):
        self._validation_generator = value

    @property
    def test_generator(self):
        return self._test_generator

    @test_generator.setter
    def test_generator(self, value):
        self._test_generator = value

    @property
    def workers(self):
        return self._workers

    @property
    def multiprocessing(self):
        return self._multiprocessing

    @property
    def gpus(self):
        return self._gpus

    @property
    def project_name(self):
        return self._project_name

    @property
    def workspace(self):
        return self._workspace

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def network(self):
        return self._network

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def train_generator(self):
        return self._train_generator

    @property
    def validation_generator(self):
        return self._validation_generator

    @property
    def test_generator(self):
        return self._test_generator

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def displayer(self):
        return self._displayer
