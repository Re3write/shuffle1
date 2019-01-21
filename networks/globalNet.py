import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from networks.group_normal import GroupNorm
from networks.Attention_Module import CAB_improve2


class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals = []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            # predict.append(self._predict(output_shape, num_class))
            # if i != len(channel_settings) - 1 or i!=0 or i!=1:
            #     upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.predict4 = self._predict(128, 24)
        # self.predict3 = self._predict(128, 24)
        # self.predict2 = self._predict(128, 24)
        self.dil6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                              padding=1, bias=False,
                              dilation=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.dil12 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                               padding=2, bias=False,
                               dilation=2)
        self.bn12 = nn.BatchNorm2d(256)

        self.dil18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=3, dilation=3)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        # self.upsamples = nn.ModuleList(upsamples)
        self.predict = self._predict(output_shape, num_class)

        # self.CAB = CAB_improve2(1024, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
                                      kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms = []
        feature1 = self.laterals[0](x[0])
        feature2 = self.laterals[1](x[1])
        feature3 = self.laterals[2](x[2])
        feature4 = self.laterals[3](x[3])

        x4 = F.relu(self.bn18((self.dil18(feature4))))
        x3 = F.relu(self.bn12(self.dil12(feature3)))
        x2 = F.relu(self.bn6(self.dil6(feature2)))

        # global_out_x4 = self.predict4(x4)
        # global_out_x3 = self.predict3(x3)
        # global_out_x2 = self.predict2(x2)

        # global_outs=self.predict4(x4)

        x3_up = nn.Upsample((128, 128), mode='bilinear', align_corners=True)(x3)
        x2_up = nn.Upsample((128, 128), mode='bilinear', align_corners=True)(x2)
        feature1_up = nn.Upsample((128, 128), mode='bilinear', align_corners=True)(feature1)

        x3_de = self.deconv3(x3)
        x2_de = self.deconv2(x2)
        feature1_de = self.deconv1(feature1)

        x3 = x3_de + x3_up
        x2 = x2_de + x2_up
        feature1 = feature1_de + feature1_up

        global_fm = torch.cat([x4, x3, x2, feature1], 1)
        global_fm = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(global_fm)))
        # global_fm = F.relu(self.bn_conv_1x1_3(self.CAB((x4,x3,x2,feature1))))

        global_outs = self.predict(global_fm)

        return global_fm, global_outs
