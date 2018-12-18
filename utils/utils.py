import os
import sys
import json
import torch
import shutil
import numpy as np
from config import config
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch.nn.init as init


# 来源kaggle human protein
# save best model
def save_checkpoint(state, is_best_loss, is_best_f1, fold, epoch):
    filename = config.weights + config.model_name + os.sep + str(fold) + os.sep + "res50checkpoint{}.pth.tar".format(
        epoch)
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,
                        "%s/%s_fold_%s_res50model_best_loss.pth.tar" % (
                        config.best_models, config.model_name, str(fold)))
    if is_best_f1:
        shutil.copyfile(filename,
                        "%s/%s_fold_%s_res50model_best_f1.pth.tar" % (config.best_models, config.model_name, str(fold)))


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        t = Variable(y).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)


    else:
        raise NotImplementedError


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

#
# def set_optimizer_lr(optimizer, lr):
#     # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer
#
# def sgdr(period, batch_idx):
#     # returns normalised anytime sgdr schedule given period and batch_idx
#     # best performing settings reported in paper are T_0 = 10, T_mult=2
#     # so always use T_mult=2
#     batch_idx = float(batch_idx)
#     restart_period = period
#     while batch_idx/restart_period > 1.:
#         batch_idx = batch_idx - restart_period
#         restart_period = restart_period * 2.
#
#     radians = math.pi*(batch_idx/restart_period)
#     return 0.5*(1.0 + math.cos(radians))
#
# lr_trace = []
#
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     global optimizer
#     start_batch_idx = len(trainloader)*epoch
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         global_step = batch_idx+start_batch_idx
#         batch_lr = args.lr*sgdr(lr_period, global_step)
#         lr_trace.append(batch_lr)
#         optimizer = set_optimizer_lr(optimizer, batch_lr)
#         optimizer.zero_grad()
#         inputs, targets = Variable(inputs), Variable(targets)
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         if len(lr_trace) > 1:
#             if lr_trace[-1] - lr_trace[-2] > 1e-3:
#                 # we've just reset the learning rate
#                 if args.sparsify:
#                     print("Sparsifying at step %i..."%global_step)
#                     optimizer.sparsify()
#                 print("Sparsitying is %f"%optimizer.sparsity())
#         optimizer.step()
#
#         train_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()
#
#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.3f'
#             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, batch_lr))
