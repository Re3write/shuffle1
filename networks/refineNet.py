import torch.nn as nn
import torch
from networks.aasp import ASPP
from networks.group_normal import GroupNorm
from networks.Attention_Module import SpatialAttention
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        # self.shortcut = nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(planes * 4)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        # self.se = SELayer(planes * 2)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.se(out)

        # residual = self.shortcut(residual)
        # residual = self.bn4(residual)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class refineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(refineNet, self).__init__()

        self.cascade = self._make_layer(lateral_channel, out_shape)
        self.final_predict = self._predict(lateral_channel, num_class)
        self.aasp = ASPP()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # self.spA = SpatialAttention(7)
        # self.bn=nn.BatchNorm2d(256)

    def _make_layer(self, input_channel, output_shape):
        layers = []
        for i in range(2):
            layers.append(Bottleneck(input_channel, 128))
        # layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(Bottleneck(input_channel, 128))
        # layers.append(SELayer(256))
        layers.append(nn.Conv2d(256, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cascade(x)
        out = self.aasp(x)
        # out = self.spA(out) * out
        # out =F.relu(self.bn(out))
        out = self.final_predict(out)
        return out
