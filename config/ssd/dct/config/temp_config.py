from os import environ
from os.path import join


from tensorflow.keras.optimizers import Adadelta, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from jpeg_deep.networks import ssd300
from jpeg_deep.generators import VOCGeneratorDCT
from jpeg_deep.losses import ssd_loss
from jpeg_deep.generators import SSDInputEncoder
from jpeg_deep.evaluation import Evaluator
from jpeg_deep.displayer import Displayer
from jpeg_deep.generators import SSDDataAugmentation
from jpeg_deep.generators import ConvertTo3Channels, Resize
from jpeg_deep.losses import SSDLoss
from jpeg_deep.displayer import DisplayerObjects
#from template.config import TemplateConfiguration


class TrainingConfiguration(object):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = "Example config file to train in the RGB domain."

        # System dependent variable
        self._workers = 14
        self._multiprocessing = True
        self._gpus = 1

        # Variables for comet.ml
        self._project_name = "ssd_jpeg"
        self._workspace = "d3lt4lph4"

        # Network variables
        self.n_classes = 20
        self.image_shape = (300, 300)
        self._weights = None
        self.network_params = {"n_classes": self.n_classes,
                               "image_shape": self.image_shape, "dct": True}
        self._network = ssd300(**self.network_params)

        # Training variables
        self._epochs = 120
        self._batch_size = 32
        self._steps_per_epoch = 1000

        # Optimizer, loss and metric
        self.optimizer_params = {
            "lr": 0.01, "momentum": 0.9, "decay": 0.0005, "nesterov": True}
        self._optimizer = SGD(**self.optimizer_params)
        self.loss_class = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self._loss = self.loss_class.compute_loss
        self._metrics = ['accuracy']

        # Generator parameters.
        self.predictor_sizes = [[38, 38], [19, 19],
                                [10, 10], [5, 5], [3, 3], [1, 1]]
        self.scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        self.aspect_ratios = [[1.0, 2.0, 0.5],
                              [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                              [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                              [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                              [1.0, 2.0, 0.5],
                              [1.0, 2.0, 0.5]]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.label_encoder_params = {
            "img_height": self.image_shape[0],
            "img_width": self.image_shape[1],
            "n_classes": self.n_classes,
            "predictor_sizes": self.predictor_sizes,
            "scales": self.scales,
            "aspect_ratios_per_layer": self.aspect_ratios,
            "steps": self.steps,
            "offsets": self.offsets,
            "pos_iou_threshold": 0.5,
            "neg_iou_limit": 0.5
        }
        self.label_encoder = SSDInputEncoder(**self.label_encoder_params)
        # The transformations
        self.ssd_data_augmentation = SSDDataAugmentation(img_height=self.image_shape[0],
                                                         img_width=self.image_shape[1],
                                                         background=[0, 0, 0])
        self.convert_to_3_channels = ConvertTo3Channels()
        self.resize = Resize(
            height=self.image_shape[0], width=self.image_shape[1])

        self.train_directory = environ["DATASET_PATH_TRAIN"]
        self.validation_directory = environ["DATASET_PATH_VAL"]
        self.test_directory = environ["DATASET_PATH_TEST"]

        self.training_generator_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "label_encoder": self.label_encoder,
            "transforms": [self.ssd_data_augmentation],
            "images_path": [join(self.train_directory, "VOC2007_trainval/ImageSets/Main/train.txt"),
                            join(self.train_directory, "VOC2012_trainval/ImageSets/Main/train.txt")]
        }

        self.validation_generator_params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "label_encoder": self.label_encoder,
            "transforms": [self.convert_to_3_channels,
                           self.resize],
            "images_path": [join(self.validation_directory, "VOC2007_trainval/ImageSets/Main/val.txt"),
                            join(self.validation_directory, "VOC2012_trainval/ImageSets/Main/val.txt")]
        }

        self.testing_generator_params = {
            "batch_size": 1,
            "shuffle": False,
            "images_path": [join(self.test_directory, "VOC2007_test/ImageSets/Main/test.txt")]
        }

        # Keras stuff
        self.model_checkpoint = None
        self.csv_logger = None
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=10)

        self._callbacks = [self.terminate_on_nan, self.early_stopping]

        # Creating the training and validation generator
        self._train_generator = VOCGeneratorDCT(
            **self.training_generator_params)
        self._validation_generator = VOCGeneratorDCT(
            **self.validation_generator_params)
        self._test_generator = VOCGeneratorDCT(
            **self.testing_generator_params)

        self._displayer = DisplayerObjects()
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
        self.network_params["mode"] = "inference"
        self.network_params["image_shape"] = None
        self._network = ssd300(**self.network_params)

    def prepare_evaluator(self):
        self._evaluator = Evaluator()

    def prepare_testing_generator(self):
        self._test_generator.prepare_dataset()

    def prepare_training_generators(self):
        self._train_generator.prepare_dataset()
        self._validation_generator.prepare_dataset()

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
