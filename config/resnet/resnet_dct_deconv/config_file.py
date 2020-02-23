from os.path import join
from os import environ

from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, TensorBoard

from jpeg_deep.generators import DCTGeneratorJPEG2DCT
from jpeg_deep.networks import deconvolution_rfa
from jpeg_deep.evaluation import Evaluator

from albumentations import (
    HorizontalFlip,
    RandomCrop,
    CenterCrop,
    SmallestMaxSize,
)

from albumentations import (
    OneOf,
    Compose
)


class TrainingConfiguration(object):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.description = "Training configuration file for the RGB version of the ResNet50 network."

        # System dependent variable
        self._workers = 10
        self._multiprocessing = True

        # Variables for comet.ml
        self._project_name = "jpeg-deep"
        self._workspace = "classification_dct"

        # Network variables
        self._weights = None
        self._network = deconvolution_rfa()

        # Training variables
        self._epochs = 140
        self._batch_size = 32
        self._steps_per_epoch = 1281167 // self.batch_size
        self._validation_steps = 50000 // self._batch_size
        self.optimizer_parameters = {
            "lr": 0.0125, "momentum": 0.9}
        self._optimizer = SGD(**self.optimizer_parameters)
        self._loss = categorical_crossentropy
        self._metrics = ['accuracy', 'top_k_categorical_accuracy']

        self.train_directory = join(
            environ["DATASET_PATH_TRAIN"], "imagenet/train")
        self.validation_directory = join(
            environ["DATASET_PATH_TRAIN"], "imagenet/validation")
        self.test_directory = join(
            environ["DATASET_PATH_VAL"], "imagenet/validation")
        self.validation_split = 0.95
        self.index_file = "/home/2017018/bdegue01/git/jpeg_deep/data/imagenet_class_index.json"

        # Defining the transformations that will be applied to the inputs.
        self.train_transformations = [
            SmallestMaxSize(256),
            RandomCrop(224, 224),
            HorizontalFlip()
        ]

        self.validation_transformations = [
            SmallestMaxSize(256), CenterCrop(224, 224)]

        # Keras stuff
        self.model_checkpoint = None
        self.reduce_lr_on_plateau = ReduceLROnPlateau(patience=5, verbose=1)
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=11)

        self._callbacks = [self.reduce_lr_on_plateau,
                           self.terminate_on_nan, self.early_stopping]

        self._train_generator = None
        self._validation_generator = None
        self._test_generator = None

        self._horovod = None

    def prepare_runtime_checkpoints(self, directories_dir):
        log_dir = directories_dir["log_dir"]
        checkpoints_dir = directories_dir["checkpoints_dir"]

        self._callbacks.append(ModelCheckpoint(filepath=join(
            checkpoints_dir,
            "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
            save_best_only=True))
        self._callbacks.append(
            TensorBoard(log_dir))

    def prepare_horovod(self, hvd):
        self._horovod = hvd
        self.optimizer_parameters["lr"] = self.optimizer_parameters["lr"] * hvd.size()
        self._optimizer = SGD(**self.optimizer_parameters)
        self._optimizer = hvd.DistributedOptimizer(self._optimizer)
        self._steps_per_epoch = self._steps_per_epoch // hvd.size()
        self._validation_steps = 3 * self._validation_steps // hvd.size()

        self._callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),

            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),

            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(
                warmup_epochs=5, verbose=1),

            # Reduce the learning rate if training plateaues.
            self.reduce_lr_on_plateau,

            self.terminate_on_nan,

            self.early_stopping
        ]

    def prepare_for_inference(self):
        pass

    def prepare_evaluator(self):
        self._evaluator = Evaluator()

    def prepare_testing_generator(self):
        self._test_generator = DCTGeneratorJPEG2DCT(
            self.test_directory, self.index_file, None, 1)

    def prepare_training_generators(self):
        self._train_generator = DCTGeneratorJPEG2DCT(
            self.train_directory, self.index_file, batch_size=self.batch_size, shuffle=True, transforms=self.train_transformations, split_cbcr=True)
        self._validation_generator = DCTGeneratorJPEG2DCT(
            self.validation_directory, self.index_file, batch_size=self.batch_size, shuffle=True, transforms=self.validation_transformations, split_cbcr=True)

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
    def batch_size(self):
        return self._batch_size

    @property
    def gpus(self):
        return self._gpus

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def loss(self):
        return self._loss

    @property
    def displayer(self):
        return self._displayer

    @property
    def metrics(self):
        return self._metrics

    @property
    def multiprocessing(self):
        return self._multiprocessing

    @property
    def network(self):
        return self._network

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def workers(self):
        return self._workers

    @property
    def epochs(self):
        return self._epochs

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def project_name(self):
        return self._project_name

    @property
    def workspace(self):
        return self._workspace

    @property
    def horovod(self):
        return self._horovod

    @property
    def validation_steps(self):
        return self._validation_steps
