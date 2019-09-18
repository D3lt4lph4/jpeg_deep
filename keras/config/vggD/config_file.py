from os.path import join

from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from vgg_jpeg_keras.networks import vggd


def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    return _func


class TrainingConfiguration(object):
    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = ""

        # System dependent variable
        self.workers = 10
        self.multiprocessing = True
        self.gpus = 1

        # Variables for comet.ml
        self.project_name = "vgg-dct"
        self.workspace = "d3lt4lph4"

        # Network variables
        self.num_classes = 1000
        self.img_size = (224, 224)
        self.weights = "/home/2017018/bdegue01/experiments/d3lt4lph4_vgg-dct_DWqlB8A2AGBxVpomEqWuNpPrggXkvMAe/checkpoints/epoch-07_loss-2.4218_val_loss-2.5476.h5"
        self.network = vggd(self.num_classes)

        # Training variables
        self.epochs = 120
        self.batch_size = 256
        self.batch_per_epoch = 5000
        self.optimizer = SGD(lr=0.01,
                             momentum=0.9,
                             decay=0.0005,
                             nesterov=True)
        self.loss = categorical_crossentropy
        self.metrics = ['accuracy']
        self.train_directory = join(
            environ["DATASET_PATH_TRAIN"], "imagenet/train")
        self.validation_directory = join(
            environ["DATASET_PATH_VAL"], "imagenet/validation")

        # Keras stuff
        self.model_checkpoint = None
        self.csv_logger = None
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=7)

        self.callbacks = [self.terminate_on_nan, self.early_stopping]

        # Creating the training and validation generator
        self.train_generator = None
        self.validation_generator = None

        self._horovod = None

    def add_csv_logger(self,
                       output_path,
                       filename="results.csv",
                       separator=',',
                       append=True):
        if self.horovod is not None:
            if self.horovod.rank() == 0:
                self.csv_logger = CSVLogger(filename=join(output_path, filename),
                                            separator=separator,
                                            append=append)
                self._callbacks.append(self.csv_logger)
        else:
            self.csv_logger = CSVLogger(filename=join(output_path, filename),
                                        separator=separator,
                                        append=append)
            self._callbacks.append(self.csv_logger)

    def add_model_checkpoint(self, output_path, verbose=1,
                             save_best_only=True):
        if self.horovod is not None:
            if self.horovod.rank() == 0:
                self._callbacks.append(ModelCheckpoint(filepath=join(
                    output_path,
                    "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                    verbose=verbose,
                    save_best_only=save_best_only))
        else:
            self.model_checkpoint = ModelCheckpoint(filepath=join(
                output_path,
                "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                verbose=verbose,
                save_best_only=save_best_only)
            self._callbacks.append(self.model_checkpoint)

    def prepare_horovod(self, hvd):
        self._horovod = hvd
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
            ReduceLROnPlateau(patience=10, verbose=1),

            self.terminate_on_nan,

            self.early_stopping
        ]

        self.optimizer_parameters["lr"] = self.optimizer_parameters["lr"] * \
            hvd.size() / self.batch_size_divider
        self._optimizer = hvd.DistributedOptimizer(self._optimizer)
        self._batch_size = self._batch_size // self.batch_size_divider
        self._steps_per_epoch = self._steps_per_epoch // (
            hvd.size() // self.batch_size_divider)
        self._validation_steps = 3 * self._validation_steps // hvd.size()

    def prepare_for_inference(self):
        pass

    def prepare_evaluator(self):
        self._evaluator = Evaluator()

    def prepare_testing_generator(self):
        pass

    def prepare_training_generators(self):
        self._train_generator = ImageDataGenerator(featurewise_center=True,
                                                   rotation_range=0.2,
                                                   shear_range=0.2,
                                                   zoom_range=0.2,
                                                   vertical_flip=True,
                                                   validation_split=0,
                                                   preprocessing_function=preprocess_input).flow_from_directory(
            self.train_directory,
            target_size=self.img_size,
            batch_size=self.batch_size)
        self._validation_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input).flow_from_directory(
                self.validation_directory,
                target_size=self.img_size,
                batch_size=self.batch_size)

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
