import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out.size())
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.size())
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        return self.sigmoid(x)


# cbam
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

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1, x2], dim=1)
        print(x.size())
        x = self.global_pooling(x)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.relu(x)
        x = self.conv2(x)
        print(x.size())
        x = self.sigmod(x)
        print(x.size())
        x2 = x * x2
        res = x2 + x1
        return res


class CAB_improve(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB_improve, self).__init__()
        self.global_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.sigmod1 = nn.Sigmoid()

        self.global_pooling2 = nn.AdaptiveAvgPool2d(1)
        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.sigmod2 = nn.Sigmoid()

        self.global_pooling3 = nn.AdaptiveAvgPool2d(1)
        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv32 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(out_channels)
        self.sigmod3 = nn.Sigmoid()

        self.global_pooling4 = nn.AdaptiveAvgPool2d(1)
        self.conv41 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu4 = nn.ReLU()
        self.bn41 = nn.BatchNorm2d(out_channels)
        self.conv42 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(out_channels)
        self.sigmod4 = nn.Sigmoid()

        self.global_pooling5 = nn.AdaptiveAvgPool2d(1)
        self.conv51 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn51 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU()
        self.conv52 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn52 = nn.BatchNorm2d(out_channels)
        self.sigmod5 = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        out1 = self.global_pooling1(x)
        out1 = self.bn11(self.conv11(out1))
        out1 = self.relu1(out1)
        out1 = self.bn12(self.conv12(out1))
        out1 = self.sigmod1(out1)

        out2 = self.global_pooling2(x)
        out2 = self.bn21(self.conv21(out2))
        out2 = self.relu2(out2)
        out2 = self.bn22(self.conv22(out2))
        out2 = self.sigmod2(out2)

        out3 = self.global_pooling3(x)
        out3 = self.bn31(self.conv31(out3))
        out3 = self.relu3(out3)
        out3 = self.bn32(self.conv32(out3))
        out3 = self.sigmod3(out3)

        out4 = self.global_pooling4(x)
        out4 = self.bn41(self.conv41(out4))
        out4 = self.relu4(out4)
        out4 = self.bn42(self.conv42(out4))
        out4 = self.sigmod4(out4)

        out5 = self.global_pooling5(x)
        out5 = self.bn51(self.conv51(out5))
        out5 = self.relu5(out5)
        out5 = self.bn52(self.conv52(out5))
        out5 = self.sigmod5(out5)

        final_output = out1 * x1 + out2 * x2 + out3 * x3 + out4 * x4 + out5 * x5
        return final_output


class CAB_improve2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB_improve2, self).__init__()
        self.global_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.sigmod1 = nn.Sigmoid()

        self.global_pooling2 = nn.AdaptiveAvgPool2d(1)
        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.sigmod2 = nn.Sigmoid()

        self.global_pooling3 = nn.AdaptiveAvgPool2d(1)
        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv32 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(out_channels)
        self.sigmod3 = nn.Sigmoid()

        self.global_pooling4 = nn.AdaptiveAvgPool2d(1)
        self.conv41 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn41 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()
        self.conv42 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(out_channels)
        self.sigmod4 = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3, x4 = x
        x = torch.cat([x1, x2, x3, x4], dim=1)

        out1 = self.global_pooling1(x)
        out1 = self.bn11(self.conv11(out1))
        out1 = self.relu1(out1)
        out1 = self.bn12(self.conv12(out1))
        out1 = self.sigmod1(out1)

        out2 = self.global_pooling2(x)
        out2 = self.bn21(self.conv21(out2))
        out2 = self.relu2(out2)
        out2 = self.bn22(self.conv22(out2))
        out2 = self.sigmod2(out2)

        out3 = self.global_pooling3(x)
        out3 = self.bn31(self.conv31(out3))
        out3 = self.relu3(out3)
        out3 = self.bn32(self.conv32(out3))
        out3 = self.sigmod3(out3)

        out4 = self.global_pooling4(x)
        out4 = self.bn41(self.conv41(out4))
        out4 = self.relu4(out4)
        out4 = self.bn42(self.conv42(out4))
        out4 = self.sigmod4(out4)

        final_output = out1 * x1 + out2 * x2 + out3 * x3 + out4 * x4
        return final_output


class cab_dense1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cab_dense1, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)

        self.global_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.sigmod1 = nn.Sigmoid()

        self.global_pooling2 = nn.AdaptiveAvgPool2d(1)
        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.sigmod2 = nn.Sigmoid()

        self.global_pooling3 = nn.AdaptiveAvgPool2d(1)
        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv32 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(out_channels)
        self.sigmod3 = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3 = x
        x1 = F.relu(self.bn(self.conv(x1)))
        x = torch.cat([x1, x2, x3], dim=1)

        out1 = self.global_pooling1(x)
        out1 = self.bn11(self.conv11(out1))
        out1 = self.relu1(out1)
        out1 = self.bn12(self.conv12(out1))
        out1 = self.sigmod1(out1)

        out2 = self.global_pooling2(x)
        out2 = self.bn21(self.conv21(out2))
        out2 = self.relu2(out2)
        out2 = self.bn22(self.conv22(out2))
        out2 = self.sigmod2(out2)

        out3 = self.global_pooling3(x)
        out3 = self.bn31(self.conv31(out3))
        out3 = self.relu3(out3)
        out3 = self.bn32(self.conv32(out3))
        out3 = self.sigmod3(out3)

        final_output = out1 * x1 + out2 * x2 + out3 * x3
        return final_output
