from os.path import join

from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from dctvgg.networks import VGG16_A

class TrainingConfiguration(object):

    def __init__(self):
        # Variables to hold the description of the experiment
        self.config_description = "This is the configuration file to train the VGG16 from scratch on the imagenet dataset. This config file is for training of the first network VGG16_A "
        self.experiment_description = "Training the VGG16_A network for the 224x224 imagenet dataset. Testing with multiple workers."
        self.experiment_name = "VGG16_A 224x224"

        # System dependent variable
        self.workers = 28
        self.multiprocessing = True 
        self.gpus = 2

        # Variables for comet.ml
        self.project_name = "vgg-dct"
        self.workspace = "d3lt4lph4"

        # Network variables
        self.num_classes = 1000
        self.img_size = (224, 224)
        self.weights = None
        self.network = VGG16_A(self.num_classes)

        # Training variables
        self.epochs = 120
        self.batch_size = 256
        self.batch_per_epoch = 5000
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=0, nesterov=True)
        self.loss = categorical_crossentropy
        self.metrics = ['accuracy']
        self.train_directory = "/save/2017018/bdegue01/imagenet/training"
        self.validation_directory = "/save/2017018/bdegue01/imagenet/validation"

        # Keras stuff
        self.model_checkpoint = None
        self.csv_logger = None
        self.save_history = None
        self.terminate_on_nan = TerminateOnNaN()
        self.early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=7)

        self.callbacks = [self.terminate_on_nan, self.early_stopping]

        # Creating the training and validation generator
        self.train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
            self.train_directory, target_size=self.img_size, batch_size=self.batch_size)
        self.validation_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
            self.validation_directory, target_size=self.img_size, batch_size=self.batch_size)

    def add_csv_logger(self, output_path, filename="results.csv", separator=',', append=True):
        self.csv_logger = CSVLogger(filename=join(
            output_path, filename), separator=separator, append=append)
        self.callbacks.append(self.csv_logger)

    def add_model_checkpoint(self, output_path, verbose=1, save_best_only=True):
        self.model_checkpoint = ModelCheckpoint(filepath=join(
            output_path, "epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"), verbose=verbose, save_best_only=save_best_only)
        self.callbacks.append(self.model_checkpoint)
