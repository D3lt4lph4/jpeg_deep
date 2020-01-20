from argparse import ArgumentParser

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

from keras.metrics import top_k_categorical_accuracy

from keras.metrics import top_k_categorical_accuracy

from albumentations import SmallestMaxSize

from jpeg_deep.generators import RGBGenerator
from jpeg_deep.networks import vggd, vggd_conv


def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    _func.__name__ = "_func_{}".format(k)
    return _func

parser = ArgumentParser()
parser.add_argument("-wp", help="The path to the weights to be tested.", type=str)
parser.add_argument("-dp", help="The path to the directory containing the evaluation classes.")
parser.add_argument("-jf", help="The json file containing the matching for the different classes/index.")
args = parser.parse_args()

transformations = [SmallestMaxSize(256)]

# Creation of the model
model = vggd_conv(1000)
model.load_weights(args.wp, by_name=True)

# Compiling the model
model.compile(optimizer=SGD(), loss=categorical_crossentropy,
              metrics=[_top_k_accuracy(1), _top_k_accuracy(5)])

# Creating the generator for the testing
generator = RGBGenerator(args.dp, args.jf, input_size=(None), batch_size=1, transforms=transformations)

# Evaluating the network
print(model.evaluate_generator(generator, verbose=1))
