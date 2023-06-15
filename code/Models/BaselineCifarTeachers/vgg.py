import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class BaselineVGG(nn.Module):
    def __init__(self, vgg_name, use_bn=True, init_scheme=None):
        super(BaselineVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        self.classifier = nn.Linear(512, 10)

        if init_scheme is not None:
            self._init_weights(init_scheme)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, use_bn):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if use_bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _init_weights(self, init_scheme):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init_scheme(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def BaselineVGG11(use_bn=True):
    return BaselineVGG('VGG11', use_bn)


def BaselineVGG13(use_bn=True):
    return BaselineVGG('VGG13', use_bn)


def BaselineVGG16(use_bn=True):
    return BaselineVGG('VGG16', use_bn)


def BaselineVGG19(use_bn=True):
    return BaselineVGG('VGG19', use_bn)
