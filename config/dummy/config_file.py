from os.path import join

from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau

from vgg_jpeg.networks import VGG16_A, VGG16_D
from vgg_jpeg.networks import VGG16A3CBNI, VGG16A3CBNIDeconvolution
from vgg_jpeg.networks import VGG16D3CBNI, VGG16D3CBNIDeconvolution
from vgg_jpeg.networks import VGG16A3CBNA
from vgg_jpeg.networks import VGG16A3CNoBN
from vgg_jpeg.generators import DummyGenerator
from vgg_jpeg.evaluation import Evaluator

from template.config import TemplateConfiguration


class TrainingConfiguration(TemplateConfiguration):
    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = "Custom training file, for testing and experimenting purpose."

        # System dependent variable
        self._workers = 1
        self._multiprocessing = False
        self._gpus = 1

        # Variables for comet.ml
        self._project_name = "vgg-dct"
        self._workspace = "d3lt4lph4"

        # Network variables
        self.num_classes = 1000
        self.img_size = (224, 224)
        self._weights = None
        self._network = VGG16D3CBNI(self.num_classes)

        # Training variables
        self._epochs = 120
        self._batch_size = 4
        self._batch_per_epoch = 5000
        self.optimizer_params = {
            "lr": 0.01,
            "momentum": 0.9,
            "decay": 0.0005,
            "nesterov": True
        }
        self._optimizer = SGD(**self.optimizer_params)
        self._loss = categorical_crossentropy
        self._metrics = ['accuracy']
        self.train_directory = "/save/2017018/bdegue01/imagenet/training"
        self.validation_directory = "/save/2017018/bdegue01/imagenet/validation"
        self.index_file = "/home/2017018/bdegue01/git/these_code_testing/vgg_jpeg/data/imagenet_class_index.json"

        # Keras stuff
        self.model_checkpoint = None
        self.csv_logger = None
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping_params = {
            "monitor": 'val_loss',
            "min_delta": 0,
            "patience": 8
        }
        self.early_stopping = EarlyStopping(**self.early_stopping_params)
        self.reduce_lr_on_plateau_params = {
            "monitor": 'val_loss',
            "factor": 0.1,
            "patience": 5
        }
        self.reduce_lr_on_plateau = ReduceLROnPlateau(
            **self.reduce_lr_on_plateau_params)

        self._callbacks = [
            self.terminate_on_nan, self.early_stopping,
            self.reduce_lr_on_plateau
        ]

        # Creating the training and validation generator
        self.num_batches = 100
        self._train_generator = DummyGenerator(self.num_batches,
                                               self.batch_size)
        self._validation_generator = DummyGenerator(self.num_batches,
                                                    self.batch_size)
        self._test_generator = DummyGenerator(self.num_batches,
                                              self.batch_size)
        self._evaluator = None

    def add_csv_logger(self,
                       output_path,
                       filename="results.csv",
                       separator=',',
                       append=True):
        self.csv_logger = CSVLogger(filename=join(output_path, filename),
                                    separator=separator,
                                    append=append)
        self.callbacks.append(self.csv_logger)

    def add_model_checkpoint(self, output_path, verbose=1,
                             save_best_only=True):
        self.model_checkpoint = ModelCheckpoint(filepath=join(
            output_path,
            "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                                                verbose=verbose,
                                                save_best_only=save_best_only)
        self.callbacks.append(self.model_checkpoint)

    def prepare_for_inference(self):
        pass

    def prepare_testing_generator(self):
        pass

    def prepare_training_generators(self):
        pass

    def prepare_evaluator(self):
        self._evaluator = Evaluator()

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
