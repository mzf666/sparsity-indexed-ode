from Models.BaselineCifarTeachers import vgg, resnet

model_dict = {
    'vgg11': vgg.BaselineVGG11(use_bn=False),
    'vgg11_bn': vgg.BaselineVGG11(use_bn=True),
    'vgg13': vgg.BaselineVGG13(use_bn=False),
    'vgg13_bn': vgg.BaselineVGG13(use_bn=True),
    'vgg16': vgg.BaselineVGG16(use_bn=False),
    'vgg16_bn': vgg.BaselineVGG16(use_bn=True),
    'vgg19': vgg.BaselineVGG19(use_bn=False),
    'vgg19_bn': vgg.BaselineVGG19(use_bn=True),

    'resnet18': resnet.ResNet18(),
    'resnet34': resnet.ResNet34(),
    'resnet50': resnet.ResNet50(),
    'resnet101': resnet.ResNet101(),
    'resnet152': resnet.ResNet152(),
}
