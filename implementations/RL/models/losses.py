import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time
import torch.nn.functional as F

import json
import os
import os.path
from collections import OrderedDict


def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var**2).sum(dim=2) + 1e-8).sqrt() # TODO try infinity norm
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result


class CrossEntropyLossSeg(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLossSeg, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        '''
        :param inputs: BxclassxN
        :param targets: BxN
        :return:
        '''
        inputs = inputs.unsqueeze(3)
        targets = targets.unsqueeze(2)
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def visualize_pc_seg(score, seg, label, visualizer, opt, input_pc, batch_num):
    # display only one instance of pc/img
    input_pc_np = input_pc.cpu().numpy().transpose()  # Nx3
    pc_color_np = np.ones(input_pc_np.shape, dtype=int)  # Nx3
    gt_pc_color_np = np.ones(input_pc_np.shape, dtype=int)  # Nx3

    # construct color map
    _, predicted_seg = torch.max(score, dim=0, keepdim=False)  # 50xN -> N
    predicted_seg_np = predicted_seg.cpu().numpy()  # N
    gt_seg_np = seg.cpu().numpy()  # N

    color_map_file = os.path.join(opt.dataroot, 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))
    color_map_np = np.fabs((np.asarray(color_map) * 255)).astype(int)  # 50x3

    for i in range(input_pc_np.shape[0]):
        pc_color_np[i] = color_map_np[predicted_seg_np[i]]
        gt_pc_color_np[i] = color_map_np[gt_seg_np[i]]
        if gt_seg_np[i] == 49:
            gt_pc_color_np[i] = np.asarray([1, 1, 1]).astype(int)

    dict = OrderedDict([('pc_colored_predicted', [input_pc_np, pc_color_np]),
                        ('pc_colored_gt', [input_pc_np, gt_pc_color_np])])

    visualizer.display_current_results(dict, 1, 1)


def compute_iou_np_array(score, seg, label, visualizer, opt, input_pc):
    part_label = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]

    _, seg_predicted = torch.max(score, dim=1)  # BxN

    iou_batch = []
    for i in range(score.size()[0]):
        iou_pc = []
        for part in part_label[label[i]]:
            gt = seg[i] == part
            predict = seg_predicted[i] == part

            intersection = (gt + predict) == 2
            union = (gt + predict) >= 1

            if union.sum() == 0:
                iou_part = 1.0
            else:
                iou_part = intersection.int().sum().item() / (union.int().sum().item() + 0.0001)

            iou_pc.append(iou_part)

        iou_batch.append(np.asarray(iou_pc).mean())

    iou_np = np.asarray(iou_batch)

    return iou_np


def compute_iou(score, seg, label, visualizer, opt, input_pc):
    '''
    :param score: BxCxN tensor
    :param seg: BxN tensor
    :return:
    '''

    part_label = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]

    _, seg_predicted = torch.max(score, dim=1)  # BxN

    iou_batch = []
    vis_flag = False
    for i in range(score.size()[0]):
        iou_pc = []
        for part in part_label[label[i]]:
            gt = seg[i] == part
            predict = seg_predicted[i] == part

            intersection = (gt + predict) == 2
            union = (gt + predict) >= 1

            # print(intersection)
            # print(union)
            # assert False

            if union.sum() == 0:
                iou_part = 1.0
            else:
                iou_part = intersection.int().sum().item() / (union.int().sum().item() + 0.0001)

            # debug to see what happened
            # if iou_part < 0.1:
            #     print(part)
            #     print('predict:')
            #     print(predict.nonzero())
            #     print('gt')
            #     print(gt.nonzero())
            #     vis_flag = True

            iou_pc.append(iou_part)

        # debug to see what happened
        if vis_flag:
            print('============')
            print(iou_pc)
            print(label[i])
            visualize_pc_seg(score[i], seg[i], label[i], visualizer, opt, input_pc[i], i)

        iou_batch.append(np.asarray(iou_pc).mean())

    iou = np.asarray(iou_batch).mean()

    return iou

class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self,x):
     #   neglog = - F.log_softmax(x,dim=0)
        # greater the value greater the chance of being real
        #probe = torch.mean(-F.log_softmax(x,dim=0))#F.softmax(x,dim=0)

      #  print(x.cpu().data.numpy())
       # print(-torch.log(x).cpu().data.numpy())
        return torch.mean(x)

class MSE(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSE,self).__init__()
        self.reduction = reduction

    def forward(self,x,y):
        mse = F.mse_loss(x, y, reduction=self.reduction)
        return mse


class Norm(nn.Module):
    def __init__(self,dims):
        super(Norm, self).__init__()
        self.dims = dims

    def forward(self,x):
        z2 = torch.norm(x, p=2)
        out = (z2 - self.dims)
        out = out*out
        return out