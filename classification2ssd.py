""" Credit to https://stackoverflow.com/questions/41161021/how-to-convert-a-dense-layer-to-an-equivalent-convolutional-layer-in-keras """
import h5py
from argparse import ArgumentParser

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

import numpy as np
from scipy.ndimage import zoom

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
            output_dim = layer.get_weights()[1].shape[0]
            W, b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1], f_dim[2], f_dim[3], output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1], f_dim[2]),
                                          strides=(1, 1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W, b])
                flattened_ipt = False

            else:
                shape = (1, 1, input_shape[1], output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1, 1),
                                          strides=(1, 1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W, b])

        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model


parser = ArgumentParser()
parser.add_argument(
    "mt", help="The type of the model to convert, for now one of vgga/vggd.", type=str)
parser.add_argument("wp", help="The weights to be converted", type=str)
args = parser.parse_args()

if args.mt == "vgga":
    model = vgga(1000)
else:
    model = vggd(1000)

model.load_weights(args.wp)

new_model = to_fully_conv(model)

new_model.save("converted.h5")


with h5py.File("converted.h5", 'a') as f:

    del f['model_weights']['dropout_1']
    del f['model_weights']['dropout_2']
    del f['model_weights']['conv2d_3']

    # Removing old names
    names = f['model_weights'].attrs["layer_names"]
    names = np.delete(names, -1)
    names = np.delete(names, -1)
    names = np.delete(names, -1)
    names[-2] = b'fc6'
    names[-1] = b'fc7'
    f['model_weights'].attrs["layer_names"] = names

    full_idx = np.arange(4096)
    np.random.shuffle(full_idx)
    idx = full_idx[:1024]

    # Rename the groups
    f['model_weights']['fc6'] = f['model_weights']['conv2d_1']
    f['model_weights']['fc7'] = f['model_weights']['conv2d_2']

    f['model_weights']['fc6']['fc6'] = f['model_weights']['conv2d_1']['conv2d_1']
    f['model_weights']['fc7']['fc7'] = f['model_weights']['conv2d_2']['conv2d_2']

    del f['model_weights']['conv2d_1']
    del f['model_weights']['conv2d_2']
    del f['model_weights']['fc6']['conv2d_1']
    del f['model_weights']['fc7']['conv2d_2']

    fc6_kernel = f['model_weights']['fc6']['fc6']['kernel:0']
    fc6_bias = f['model_weights']['fc6']['fc6']['bias:0']
    fc7_kernel = f['model_weights']['fc7']['fc7']['kernel:0']
    fc7_bias = f['model_weights']['fc7']['fc7']['bias:0']

    fc6_numpy = fc6_kernel.value
    fc7_numpy = fc7_kernel.value

    fc6_bias_numpy = fc6_bias.value
    fc7_bias_numpy = fc7_bias.value

    del f['model_weights']['fc6']['fc6']['kernel:0']
    del f['model_weights']['fc7']['fc7']['kernel:0']
    del f['model_weights']['fc6']['fc6']['bias:0']
    del f['model_weights']['fc7']['fc7']['bias:0']

    # Select 1024 randomly for fc6
    fc6_numpy = fc6_numpy[:, :, :, idx]
    fc6_bias_numpy = fc6_bias_numpy[idx]
    fc6_numpy = zoom(fc6_numpy, (0.42857, 0.42857, 1, 1))

    # Select 1024 randomly for fc7
    fc7_numpy = fc7_numpy[:, :, :, idx]
    fc7_bias_numpy = fc7_bias_numpy[idx]
    fc7_numpy = fc7_numpy[:, :, idx, :]

    # Reset the new kernels
    f.create_dataset('model_weights/fc6/fc6/kernel:0', data=fc6_numpy)
    f.create_dataset('model_weights/fc7/fc7/kernel:0', data=fc7_numpy)
    f.create_dataset('model_weights/fc6/fc6/bias:0', data=fc6_bias_numpy)
    f.create_dataset('model_weights/fc7/fc7/bias:0', data=fc7_bias_numpy)

    fc7_names = f['model_weights']['fc7'].attrs['weight_names']
    fc7_names[0] = b'fc7/kernel:0'
    fc7_names[1] = b'fc7/bias:0'

    f['model_weights']['fc7'].attrs['weight_names'] = fc7_names

    fc6_names = f['model_weights']['fc6'].attrs['weight_names']
    fc6_names[0] = b'fc6/kernel:0'
    fc6_names[1] = b'fc6/bias:0'

    f['model_weights']['fc6'].attrs['weight_names'] = fc6_names
