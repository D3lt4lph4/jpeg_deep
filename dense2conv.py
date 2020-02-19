""" Credit to https://stackoverflow.com/questions/41161021/how-to-convert-a-dense-layer-to-an-equivalent-convolutional-layer-in-keras """
from argparse import ArgumentParser

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

from jpeg_deep.networks import vgga, vggd


def to_fully_conv(model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model

parser = ArgumentParser()
parser.add_argument("mt", help="The type of the model to convert, for now one of vgga/vggd.", type=str)
parser.add_argument("wp", help="The weights to be converted", type=str)
args = parser.parse_args()

if args.mt == "vgga":
    model = vgga(1000)
else:
    model = vggd(1000)

model.load_weights(args.wp)

new_model = to_fully_conv(model)

new_model.save("converted.h5")