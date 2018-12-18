import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.se = se_block.SELayer(planes * 4)

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


BatchNorm = nn.BatchNorm2d


class Bottleneck_dr(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck_dr, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.se = se_block.SELayer(planes * 4)

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Shuffle_Bottleneck_dr_basic(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None,  ##inplanes = outplanes
                 dilation=(1, 1), residual=True, groups=2, c_tag=0.5):
        super(Shuffle_Bottleneck_dr_basic, self).__init__()

        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part

        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, stride=stride,
                               padding=dilation, bias=False,
                               dilation=dilation, groups=self.right_part_out)
        self.bn2 = BatchNorm(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.right_part_out)
        self.activation = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        # self.se = se_block.SELayer(planes * 4)

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)


class Shuffle_Bottleneck_dr_downsample(nn.Module):
    expansion = 4

    def __init__(self, inplanes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True,groups=2):
        super(Shuffle_Bottleneck_dr_downsample, self).__init__()
        self.conv1r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1r = BatchNorm(inplanes)
        self.conv2r = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2,
                               padding=dilation, bias=False,
                               dilation=dilation,groups=inplanes)
        self.bn2r = BatchNorm(inplanes)
        self.conv3r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3r = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride
        self.groups=groups

        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn1l = nn.BatchNorm2d(inplanes)
        self.conv2l = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(inplanes)
        # self.se = se_block.SELayer(planes * 4)

    def forward(self, x):
        out_r = self.conv1r(x)
        out_r = self.bn1r(out_r)
        out_r = self.relu(out_r)

        out_r = self.conv2r(out_r)
        out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        out_r = self.bn3r(out_r)
        out_r = self.relu(out_r)

        out_l = self.conv1l(x)
        out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        out_l = self.bn2l(out_l)
        out_l = self.relu(out_l)

        return channel_shuffle(torch.cat((out_r, out_l), 1), self.groups)


# class conv_d(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  dilation=(1, 1), residual=True):
#         super(conv_d, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = BatchNorm(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=dilation[1], bias=False,
#                                dilation=dilation[1])
#         self.bn2 = BatchNorm(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = BatchNorm(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.shortcut = nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=1, bias=False)
#         self.bn4 = BatchNorm(planes * 4)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         residual = self.shortcut(residual)
#         residual = self.bn4(residual)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

class Shuffle_ResNet(nn.Module):
    def __init__(self,block,Shuffle_block1,Shuffle_block2, layers, num_classes=1000):
        self.inplanes = 64
        super(Shuffle_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer_dr(Shuffle_block1,Shuffle_block2, 128, layers[1], stride=1, dilation=[1, 2, 5, 9])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer_dr(Shuffle_block1,Shuffle_block2, 256, layers[2], stride=1, dilation=[2, 2, 1, 2, 5, 9])
        # self.layer3 = self._make_layer_dr(block2, 256, layers[2], stride=1,dilation=[2,2,5,9,1,2,5,9,1,2,5,9,1,2,5,9,1,2,5,9,1,2,5])
        self.layer4 = self._make_layer_dr(Shuffle_block1,Shuffle_block2, 512, layers[3], stride=1, dilation=[5, 9, 17])
        # 5,9,17

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_dr(self, Shuffle_block1,Shuffle_block2, planes, blocks, stride=1, dilation=None):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        layers = []
        layers.append(Shuffle_block2(self.inplanes, stride, dilation=(dilation[0], dilation[0])))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(Shuffle_block1(self.inplanes, self.inplanes, dilation=(dilation[i], dilation[i])))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1]

class ResNet(nn.Module):
    def __init__(self, block, block2, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer2 = self._make_layer_dr(block2, 128, layers[1], stride=1, dilation=[1, 2, 5, 9])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer_dr(block2, 256, layers[2], stride=1, dilation=[2, 2, 1, 2, 5, 9])
        # self.layer3 = self._make_layer_dr(block2, 256, layers[2], stride=1,dilation=[2,2,5,9,1,2,5,9,1,2,5,9,1,2,5,9,1,2,5,9,1,2,5])
        self.layer4 = self._make_layer_dr(block2, 512, layers[3], stride=1, dilation=[5, 9, 17])
        # 5,9,17

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_dr(self, block, planes, blocks, stride=1, dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=(dilation[0], dilation[0])))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation[i], dilation[i])))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, Bottleneck_dr, [2, 2, 2, 2], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, Bottleneck_dr, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, Bottleneck_dr, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('Initialize with pre-trained ResNet')
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        print('successfully load ' + str(len(state_dict.keys())) + ' keys')
        model.load_state_dict(state_dict)
    return model


def Shuffle_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Shuffle_ResNet(Bottleneck, Shuffle_Bottleneck_dr_basic,Shuffle_Bottleneck_dr_downsample, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     print('Initialize with pre-trained ResNet')
    #     from collections import OrderedDict
    #     state_dict = model.state_dict()
    #     pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    #     for k, v in pretrained_state_dict.items():
    #         if k not in state_dict:
    #             continue
    #         state_dict[k] = v
    #     print('successfully load ' + str(len(state_dict.keys())) + ' keys')
    #     model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, Bottleneck_dr, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print('Initialize with pre-trained ResNet')
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        print('successfully load ' + str(len(state_dict.keys())) + ' keys')
        model.load_state_dict(state_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, Bottleneck_dr, [3, 8, 36, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model
