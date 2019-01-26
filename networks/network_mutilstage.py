# from .resnet import *
# from .Shuffle_resnet_dr import Shuffle_resnet50
# from .resnet_dr import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet, refineNet_withoutfinal_out
from .dense_resnet_dr import *
from .resnet2 import *

__all__ = ['CPN50', 'CPN101']


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet1 = resnet
        self.global_net1 = globalNet(channel_settings, output_shape, num_class)
        self.refine_net1 = refineNet_withoutfinal_out(channel_settings[-1], output_shape, num_class)
        self.resnet2 = resnet250
        self.global_net2 = globalNet(channel_settings, output_shape, num_class)
        self.refine_net2 = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out1 = self.resnet1(x)
        global_fms1, global_outs1 = self.global_net1(res_out1)
        refine_out1, refine_out_result = self.refine_net1(global_fms1)
        res_out2 = self.resnet2(refine_out1)
        global_fms2, global_outs2 = self.global_net2(res_out2)
        refine_out2 = self.refine_net1(global_fms2)

        return global_outs1, global_outs2, refine_out_result, refine_out2


def CPN50(out_size, num_class, pretrained=True):
    res50 = resnet50(pretrained=True)
    # res50=Shuffle_resnet50()
    model = CPN(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model


def CPN101(out_size, num_class, pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size, num_class=num_class, pretrained=pretrained)
    return model
