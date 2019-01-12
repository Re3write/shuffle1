import torch
import math
import torchvision.utils as vutils


def concact_features(conv_output):
    """
    对特征图进行reshape拼接
    :param conv_output:输入多通道的特征图
    :return:
    """
    conv_output = conv_output.permute(0, 2, 3, 1)
    num_or_size_splits = conv_output.shape[-1]
    # print("conv-output",conv_output.shape)
    each_convs = torch.split(conv_output, 1, 3)
    concact_size = int(math.sqrt(num_or_size_splits) / 1)
    # print("concact",concact_size)
    all_concact = None
    for i in range(concact_size):
        row_concact = each_convs[i * concact_size]
        for j in range(concact_size - 1):
            row_concact = torch.cat([row_concact, each_convs[i * concact_size + j + 1]], 1)
        if i == 0:
            all_concact = row_concact
        else:
            all_concact = torch.cat([all_concact, row_concact], 2)
    print(all_concact.shape)
    return all_concact


def concact_features1(conv_output):
    """
    对特征图进行reshape拼接
    :param conv_output:输入多通道的特征图
    :return:
    """
    conv_output_batch = conv_output[0].unsqueeze(1)
    all_concact = vutils.make_grid(conv_output_batch, normalize=True)

    return all_concact
