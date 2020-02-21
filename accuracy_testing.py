from argparse import ArgumentParser

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

from keras.metrics import top_k_categorical_accuracy

from keras.metrics import top_k_categorical_accuracy

from albumentations import SmallestMaxSize

from jpeg_deep.generators import RGBGenerator
from jpeg_deep.generators import DCTGeneratorJPEG2DCT

from jpeg_deep.networks import vgga_conv, vggd_conv
from jpeg_deep.networks import vgga_dct_conv, vggd_dct_conv
from jpeg_deep.networks import ResNet50, late_concat_rfa, late_concat_rfa_thinner


def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    _func.__name__ = "_func_{}".format(k)
    return _func


parser = ArgumentParser()
parser.add_argument(
    "-wp", help="The path to the weights to be tested.", type=str)
parser.add_argument(
    "-dp", help="The path to the directory containing the evaluation classes.")
parser.add_argument(
    "-jf", help="The json file containing the matching for the different classes/index.")
parser.add_argument(
    "-n", help="The type of network to load, one of vgga, vggd, vggadct, vggddct, resnet.")
args = parser.parse_args()

transformations = [SmallestMaxSize(256)]

# Creation of the model
if args.n == "vgga":
    model = vgga_conv(1000)
elif args.n == "vggd":
    model = vggd_conv(1000)
elif args.n == "vggadct":
    model = vgga_dct_conv(1000)
elif args.n == "vggddct":
    model = vggd_dct_conv(1000)
elif args.n == "resnet":
    model = ResNet50(1000)
elif args.n == "lcraf":
    model = late_concat_rfa(input_shape=None)
elif args.n == "lcraft":
    model = late_concat_rfa_thinner(input_shape=None)

model.load_weights(args.wp, by_name=True)

# Compiling the model
model.compile(optimizer=SGD(), loss=categorical_crossentropy,
              metrics=[_top_k_accuracy(1), _top_k_accuracy(5)])

# Creating the generator for the testing
if args.n in ["vggadct", "vggddct"]:
    generator = DCTGeneratorJPEG2DCT(args.dp, args.jf, input_size=(
        None), batch_size=1, transforms=transformations)
else:
    generator = RGBGenerator(args.dp, args.jf, input_size=(
        None), batch_size=1, transforms=transformations)

print(len(generator))

# Evaluating the network
print(model.evaluate_generator(generator, verbose=0))
