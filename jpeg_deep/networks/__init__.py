from .vgg import VGG16, VGG16_conv
from .networks_dct import VGG16_dct, VGG16_dct_deconv, VGG16_dct_y
from .networks_dct import VGG16_dct_conv, VGG16_dct_deconv_conv, VGG16_dct_y_conv

from .resnet import ResNet50
from .resnet_dct import late_concat_rfa, late_concat_rfa_thinner
from .resnet_dct import deconvolution_rfa
from .resnet_dct import late_concat_rfa_y, late_concat_rfa_y_thinner

from .ssd import SSD300
from .ssd_resnet import SSD300_resnet
