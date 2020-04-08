from os.path import join
from os import environ

from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

from jpeg_deep.networks import ResNet50
from jpeg_deep.evaluation import Evaluator


class TrainingConfiguration(object):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.description = "Training configuration file for the RGB version of the ResNet50 network."

        # System dependent variable
        self._workers = 10
        self._multiprocessing = True

        # Variables for comet.ml
        self._project_name = "jpeg-deep"
        self._workspace = "classification_rgb"

        # Network variables
        self._weights = None
        self._network = ResNet50()

        # Training variables
        self._epochs = 90
        self._steps_per_epoch = None
        self._validation_steps = None
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

        self._callbacks = []

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
            hvd.callbacks.LearningRateScheduleCallback(
                start_epoch=5, end_epoch=30, multiplier=1.),
            hvd.callbacks.LearningRateScheduleCallback(
                start_epoch=30, end_epoch=60, multiplier=1e-1),
            hvd.callbacks.LearningRateScheduleCallback(
                start_epoch=60, end_epoch=80, multiplier=1e-2),
            hvd.callbacks.LearningRateScheduleCallback(
                start_epoch=80, multiplier=1e-3),
        ]

    def prepare_for_inference(self):
        pass

    def prepare_evaluator(self):
        self._evaluator = Evaluator()

    def prepare_testing_generator(self):
        test_gen = ImageDataGenerator(zoom_range=(
            0.875, 0.875), preprocessing_function=preprocess_input)
        self._test_generator = test_gen.flow_from_directory(
            self.validation_directory, batch_size=self.batch_size, target_size=(224, 224))

    def prepare_training_generators(self):
        train_gen = ImageDataGenerator(width_shift_range=0.33, height_shift_range=0.33,
                                       zoom_range=0.5, horizontal_flip=True, preprocessing_function=preprocess_input)
        self._train_generator = train_gen.flow_from_directory(self.train_directory,
                                                              batch_size=self.batch_size,
                                                              target_size=(224, 224))
        val_gen = ImageDataGenerator(zoom_range=(
            0.875, 0.875), preprocessing_function=preprocess_input)
        self._validation_generator = val_gen.flow_from_directory(
            self.validation_directory, batch_size=self.batch_size, target_size=(224, 224))

        self._steps_per_epoch = len(self._train_generator)
        self._validation_steps = len(self._validation_generator)

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
