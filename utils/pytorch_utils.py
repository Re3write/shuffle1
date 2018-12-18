import torch
import torch.nn as nn
import math
from torch.optim import SGD
import torchvision
import numpy as np
from torch.nn import init
from scipy.misc import imread
import os


def freeze(model=None):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.add_block.parameters():
        param.requires_grad = True

    # ignored_params = list(map(id, model.add_block.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),  # 记住一定要加上filter()，不然会报错
        lr=0.01,
        weight_decay=1e-5, momentum=0.9, nesterov=True)

    return optimizer


def different_lr(model=None):
    ignored_params = list(map(id, model.add_block.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = SGD(
        [{'params': base_params, 'lr': 0.01},
         {'params': model.add_block.parameters(), 'lr': 0.1}],
        weight_decay=1e-5, momentum=0.9, nesterov=True)

    # optimizer_ft = optim.SGD([
    #     {'params': base_params, 'lr': 0.01},
    #     {'params': model.ViewModel.viewclassifier.parameters(), 'lr': 0.001},
    #     {'params': model.Block.parameters(), 'lr': 0.01}
    # ], weight_decay=1e-3, momentum=0.9, nesterov=True)   直接设置也行


    return optimizer


def adjust_lr_group_params(epoch,optimizer):
    step_size = 60
    base_lr=0.1
    lr = base_lr * (0.1 ** (epoch // 30))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr')
    ######################################
    ###  optimizer.param_groups 类型与内容
    # [
    #     {'params': base_params, 'lr': 0.01, 'momentum': 0.9, 'dampening': 0,
    #      'weight_decay': 0.001, 'nesterov': True, 'initial_lr': 0.01},
    #     {'params': model.ViewModel.viewclassifier.parameters(), 'lr': 0.001,
    #      'momentum': 0.9, 'dampening': 0, 'weight_decay': 0.001, 'nesterov': True,
    #      'initial_lr': 0.001},
    #     {'params': model.Block.parameters(), 'lr': 0.03, 'momentum': 0.9,
    #      'dampening': 0, 'weight_decay': 0.001, 'nesterov': True, 'initial_lr':
    #          0.03}
    #
    # ]
    ###  optimizer.param_groups 类型与内容
    ######################################

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias:
                init.constant(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1.0)
            init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=1e-3)
            init.kaiming_normal(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias:
                init.constant(m.bias, 0.0)


def layer_weights_init(m):
    # model.apply
    # 使用isinstance来判断m属于什么类型
    classname = m.__class__.__name__

    if isinstance(m, nn.Conv2d):
        print(classname)
        if m.bias not in locals().keys():
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
        else:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            init.constant_(m.bias.data, 0.0)

    elif isinstance(m, nn.BatchNorm2d):
        # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        print(classname)

    # elif isinstance(m, nn.Linear):
    #     init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
    #     init.constant_(m.bias.data, 0.0)


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
            scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=4000)
             for epoch in range(0, config.epochs):
                 print("learning rata is " + str(lr))
                    if epoch in [4, 8, 12, 16]:
                        base_lr = base_lr / 4
                        max_lr = max_lr / 4
                        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=4000)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError('{} is not an Optimizer'.format(
        #         type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


def remove_module():
    # original saved file with DataParallel
    state_dict = torch.load('myfile.pth.tar')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


if __name__ == '__main__':
    class test(nn.Module):
        def __init__(self):
            super(test, self).__init__()
            self.cov1=nn.Conv2d(3,6,kernel_size=1)


        def foward(self,x):
            x=self.cov1(x)
            x2=nn.Conv2d(6,6,kernel_size=1)(x)

            return x2


    model=test()
    def pri(m):
        print(m)
    model.apply(pri)
