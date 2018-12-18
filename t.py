# import os
#
# rootdir='D:\workplace\常远\常远测试集'
# picdir=os.listdir(rootdir)
#
# ori=os.listdir('D:\workplace\常远\常远\测试集')
# final=[]
#
# for dir in picdir:
#     tempdir_path=os.path.join(rootdir,dir)
#     tempdir=os.listdir(tempdir_path)
#     for pic in tempdir:
#         if ori.count(pic)==1:
#             final.append(pic)
#         else:
#             print(pic)
#
# print(len(final))

import numpy as np
from torch import nn
import torch

# m = nn.AdaptiveAvgPool2d(1)
input = torch.randn(4, 128, 17, 17)
# output = input.view(4, 2, 128 // 2, 17, 17).permute(0, 2, 1, 3, 4).contiguous().view(4, 128, 17, 17)
class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        #直接就是mxnet实现的permutation操作
        # def permutation(data, groups):
            #举例说明：当groups = 2时，输入：nx144x56x56
        #     data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
        #            输出：nx2x72x56x56
        #     data = mx.sym.swapaxes(data, 1, 2)
        #            输出：nx72x2x56x56
        #     data = mx.sym.reshape(data, shape=(0, -3, -2))
        #            输出：nx144x56x56
        #     return data
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        print(output.size())
        return output


x=PermutationBlock(groups=2)
input=x(input)
print(input)