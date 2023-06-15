from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .mobilenetv2 import mobile_half
from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resNet50': ResNet50,
    'wide_resnet16_1': wrn_16_1,
    'wide_resnet16_2': wrn_16_2,
    'wide_resnet40_1': wrn_40_1,
    'wide_resnet40_2': wrn_40_2,
    'vgg8_bn': vgg8_bn,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
