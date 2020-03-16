from os import environ
from os.path import join

from keras import backend as K
from keras.optimizers import Adadelta, SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from jpeg_deep.networks import SSD300
from jpeg_deep.generators import COCOGenerator
from jpeg_deep.evaluation import Evaluator

from jpeg_deep.generators import SSDInputEncoder
from jpeg_deep.tranformations import SSDDataAugmentation, ConvertTo3Channels, Resize
from jpeg_deep.losses import SSDLoss

#from template.config import TemplateConfiguration


class TrainingConfiguration(object):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = "This is the template config file."

        # System dependent variable
        self._workers = 5
        self._multiprocessing = True

        # Variables for comet.ml
        self._project_name = "jpeg_deep"
        self._workspace = "ssd"

        # Network variables
        self._weights = "/dlocal/home/2017018/bdegue01/weights/jpeg_deep/reproduce/vgg/full_reg/vggd/epoch-86_loss-1.4413_val_loss-1.9857_ssd.h5"
        self._network = SSD300(n_classes=80, scales=[
                               0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], backbone="VGG16")

        # Training variables
        self._epochs = 240
        self._batch_size = 32
        self._steps_per_epoch = 3700
        self._validation_steps = 156
        self.optimizer_parameters = {
            "lr": 0.001, "momentum": 0.9}
        self._optimizer = SGD(**self.optimizer_parameters)
        self._loss = SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss
        self._metrics = None
        dataset_path = environ["DATASET_PATH"]
        self.train_image_dir = join(dataset_path, "train2017")
        self.train_annotation_path = join(
            dataset_path, "annotations/instances_train2017.json")
        self.validation_image_dir = join(dataset_path, "val2017")
        self.validation_annotation_path = join(
            dataset_path, "annotations/instances_val2017.json")

        # Keras stuff
        self.model_checkpoint = None
        self.reduce_lr_on_plateau = ReduceLROnPlateau(patience=5, verbose=1)
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=15)

        self._callbacks = [self.reduce_lr_on_plateau, self.early_stopping,
                           self.terminate_on_nan]

        self.input_encoder = SSDInputEncoder(
            n_classes=80, scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])

        self.train_tranformations = [SSDDataAugmentation()]
        self.validation_transformations = [
            ConvertTo3Channels(), Resize(height=300, width=300)]
        self.test_transformations = [ConvertTo3Channels(), Resize(
            height=300, width=300)]

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
        K.clear_session()
        self._network = SSD300(n_classes=80, scales=[
                               0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], backbone="VGG16", mode="inference")

    def prepare_evaluator(self):
        self._evaluator = CocoEvaluator(
            self.validation_annotation_path, set="val2017", alg="ssd")

    def prepare_testing_generator(self):
        self._test_generator = COCOGenerator(self.validation_image_dir, self.validation_annotation_path, batch_size=self.batch_size, shuffle=False, label_encoder=self.input_encoder,
                                             transforms=self.test_transformations)

    def prepare_training_generators(self):
        self._train_generator = COCOGenerator(self.train_image_dir, self.train_annotation_path, batch_size=self.batch_size, shuffle=True, label_encoder=self.input_encoder,
                                              transforms=self.train_tranformations)
        self._validation_generator = COCOGenerator(self.validation_image_dir, self.validation_annotation_path, batch_size=self.batch_size, shuffle=True, label_encoder=self.input_encoder,
                                                   transforms=self.validation_transformations)
        self.validation_steps = len(self._validation_generator)

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
