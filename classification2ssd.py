""" Credit to https://stackoverflow.com/questions/41161021/how-to-convert-a-dense-layer-to-an-equivalent-convolutional-layer-in-keras """
import h5py

from os.path import split, join, splitext

from argparse import ArgumentParser

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

import numpy as np
from scipy.ndimage import zoom

from jpeg_deep.networks import vgga, vggd
from jpeg_deep.networks import vgga_dct, vggd_dct, vggd_dct_deconv
from jpeg_deep.networks import vgga_dct_conv, vggd_dct_conv, vggd_dct_deconv_conv


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


def to_fully_conv_dct(model, model_type):

    if model_type == "vgga":
        new_model = vgga_dct_conv()
    elif model_type == "vggd":
        new_model = vggd_dct_conv()
    else:
        new_model = vggd_dct_deconv_conv()

    for layer in model.layers:
        if "Dense" not in str(layer) and "Flatten" not in str(layer):
            for layer_new in new_model.layers:
                if layer_new.name == layer.name:
                    print("Setting layer : {}".format(layer_new.name))
                    layer_new.set_weights(layer.get_weights())
                    break
        elif "Dense" in str(layer):
            for layer_new in new_model.layers:
                if layer_new.name == "conv2d_1" and layer.name == "fc1":
                    print("Setting layer : {}".format(layer_new.name))
                    W, b = layer.get_weights()
                    new_W = W.reshape((7, 7, 512, 4096))
                    layer_new.set_weights([new_W, b])
                    break
                elif layer_new.name == "conv2d_2" and layer.name == "fc2":
                    print("Setting layer : {}".format(layer_new.name))
                    W, b = layer.get_weights()
                    new_W = W.reshape((1, 1, 4096, 4096))
                    layer_new.set_weights([new_W, b])
                    break
                elif layer_new.name == "conv2d_3" and layer.name == "predictions":
                    print("Setting layer : {}".format(layer_new.name))
                    W, b = layer.get_weights()
                    new_W = W.reshape((1, 1, 4096, 1000))
                    layer_new.set_weights([new_W, b])
                    break

    return new_model


parser = ArgumentParser()
parser.add_argument(
    "mt", help="The type of the model to convert, for now one of vgga/vggd.", type=str)
parser.add_argument("wp", help="The weights to be converted", type=str)
parser.add_argument("-dct", action="store_true")
args = parser.parse_args()

flag = False
if args.mt == "vgga":
    if args.dct:
        model = vgga_dct(1000)
    else:
        model = vgga(1000)
elif args.mt == "vggd":
    if args.dct:
        model = vggd_dct(1000)
    else:
        model = vggd(1000)
else:
    flag = True
    model = vggd_dct_deconv(1000)

model.load_weights(args.wp)

if args.dct or flag:
    new_model = to_fully_conv_dct(model, args.mt)
else:
    new_model = to_fully_conv(model)

# Extract saved name and resave a fully conv
path, file_name = split(args.wp)

file_name_conv = join(path, splitext(file_name)[0] + "_conv.h5")
file_name_ssd = join(path, splitext(file_name)[0] + "_ssd.h5")

new_model.save(file_name_conv)

# Also save as SSD ready
new_model.save(file_name_ssd)

with h5py.File(file_name_ssd, 'a') as f:

    if not (args.dct or flag):
        del f['model_weights']['dropout_1']
        del f['model_weights']['dropout_2']
    del f['model_weights']['conv2d_3']

    # Removing old names
    names = f['model_weights'].attrs["layer_names"]
    if not args.dct:
        idx = np.where(names == b'dropout_1')
        names = np.delete(names, idx[0])
        idx = np.where(names == b'dropout_2')
        names = np.delete(names, idx[0])
    idx = np.where(names == b'conv2d_3')
    names = np.delete(names, idx[0])

    idx = np.where(names == b'conv2d_1')
    names[idx] = b'fc6'
    idx = np.where(names == b'conv2d_2')
    names[idx] = b'fc7'
    f['model_weights'].attrs["layer_names"] = names

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
    fc6_numpy = fc6_numpy[:, :, :, ::4]
    fc6_bias_numpy = fc6_bias_numpy[::4]
    fc6_numpy = zoom(fc6_numpy, (0.42857, 0.42857, 1, 1))

    # Select 1024 randomly for fc7
    fc7_numpy = fc7_numpy[:, :, :, ::4]
    fc7_bias_numpy = fc7_bias_numpy[::4]
    fc7_numpy = fc7_numpy[:, :, ::4, :]

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
