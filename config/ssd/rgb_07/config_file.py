from os import environ
from os.path import join

from keras.optimizers import Adadelta, SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from jpeg_deep.networks import SSD300
from jpeg_deep.generators import VOCGenerator
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
        self._workers = 1
        self._multiprocessing = False

        # Variables for comet.ml
        self._project_name = "jpeg_deep"
        self._workspace = "ssd"

        # Network variables
        self._weights = "/d2/thesis/weights/keras/vgg/epoch-86_loss-1.4413_val_loss-1.9857.h5"

        # Training variables
        self._epochs = 240
        self._batch_size = 22
        self._steps_per_epoch = 1000
        self.optimizer_params = {
            "lr": 0.001, "momentum": 0.9, "decay": 0.0, "nesterov": False}
        self._optimizer = SGD(**self.optimizer_params)
        self._loss = SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss
        self._metrics = None
        dataset_path = environ["DATASET_PATH"]
        images_2007_path = join(dataset_path, "VOC2012_trainval/JPEGImages")
        self.train_sets = [(images_2007_path, join(dataset_path, "VOC2012_trainval/ImageSets/Main/train.txt"))
                           ]
        self.validation_sets = [(images_2007_path, join(
            dataset_path, "VOC2012_trainval/ImageSets/Main/val.txt"))]
        self.test_sets = [(images_2007_path, join(
            dataset_path, "VOC2012/ImageSets/Main/test.txt"))]

        # Keras stuff
        self.model_checkpoint = None
        self.reduce_lr_on_plateau = ReduceLROnPlateau(patience=7, verbose=1)
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=15)

        self._callbacks = [self.reduce_lr_on_plateau,
                           self.terminate_on_nan]

        # Creating the objects for the generators
        # Temporary variables before intergration to code
        img_height = 300
        img_width = 300
        n_classes = 20
        predictor_sizes = [[38, 38], [19, 19],
                           [10, 10], [5, 5], [3, 3], [1, 1]]
        scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                   [1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5]]
        two_boxes_for_ar1 = True
        # The space between two adjacent anchor box center points for each predictor layer.
        steps = [8, 16, 32, 64, 100, 300]
        # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
        offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
        clip_boxes = False
        # The variances by which the encoded target coordinates are divided as in the original implementation
        variances = [0.1, 0.1, 0.2, 0.2]
        normalize_coords = True
        mean_color = [123, 117, 104]
        swap_channels = [2, 1, 0]
        self._network = SSD300()
        self.input_encoder = SSDInputEncoder(img_height=img_height,
                                             img_width=img_width,
                                             n_classes=n_classes,
                                             predictor_sizes=predictor_sizes,
                                             scales=scales,
                                             aspect_ratios_per_layer=aspect_ratios_per_layer,
                                             two_boxes_for_ar1=two_boxes_for_ar1,
                                             steps=steps,
                                             offsets=offsets,
                                             clip_boxes=clip_boxes,
                                             variances=variances,
                                             matching_type='multi',
                                             pos_iou_threshold=0.5,
                                             neg_iou_limit=0.5,
                                             normalize_coords=normalize_coords)

        # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.

        self.train_tranformations = [SSDDataAugmentation(img_height=img_height,
                                                         img_width=img_width,
                                                         background=mean_color)]
        self.validation_transformations = [
            ConvertTo3Channels(), Resize(height=img_height, width=img_width)]
        self.test_transformations = [ConvertTo3Channels(), Resize(
            height=img_height, width=img_width)]

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
        print("setting hvd...........")
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
        self._test_generator = VOCGenerator(batch_size=self.batch_size, shuffle=False, label_encoder=self.input_encoder,
                                            transforms=self.test_transformations, load_images_into_memory=None, images_path=self.test_sets)

    def prepare_training_generators(self):
        self._train_generator = VOCGenerator(batch_size=self.batch_size, shuffle=True, label_encoder=self.input_encoder,
                                             transforms=self.train_tranformations, load_images_into_memory=None, images_path=self.train_sets)
        self._train_generator.prepare_dataset()
        self._validation_generator = VOCGenerator(batch_size=self.batch_size, shuffle=True, label_encoder=self.input_encoder,
                                                  transforms=self.validation_transformations, load_images_into_memory=None, images_path=self.validation_sets)
        self._validation_generator.prepare_dataset()
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
