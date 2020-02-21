from argparse import ArgumentParser

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

from albumentations import SmallestMaxSize

from jpeg_deep.generators import RGBGenerator
from jpeg_deep.generators import DCTGeneratorJPEG2DCT

from jpeg_deep.networks import vgga_conv, vggd_conv
from jpeg_deep.networks import vgga_dct_conv, vggd_dct_conv
from jpeg_deep.networks import ResNet50, late_concat_rfa, late_concat_rfa_thinner


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
    model = ResNet50(1000, (None, None))
elif args.n == "lcraf":
    model = late_concat_rfa(input_shape=None)
elif args.n == "lcraft":
    model = late_concat_rfa_thinner(input_shape=None)

model.load_weights(args.wp, by_name=True)

# Compiling the model
model.compile(optimizer=SGD(), loss=categorical_crossentropy,
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# Creating the generator for the testing
if args.n in ["vggadct", "vggddct", "lcraf", "lcraft"]:
    generator = DCTGeneratorJPEG2DCT(args.dp, args.jf, input_size=(
        None), batch_size=1, transforms=transformations)
else:
    generator = RGBGenerator(args.dp, args.jf, input_size=(
        None), batch_size=1, transforms=transformations)

print(len(generator))

# Evaluating the network
print(model.evaluate_generator(generator, verbose=0))
