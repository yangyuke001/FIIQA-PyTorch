from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FIIQALoss(nn.Module):
    def __init__(self):
        super(AGLoss, self).__init__()

    def forward(self, fiiqa_preds, fiiqa_targets):
        '''Compute loss (fiiqa_preds, fiiqa_targets) .

        Args:
          fiiqa_preds: (tensor) predicted fiiqa, sized [batch_size,100].
          fiiqa_targets: (tensor) target fiiqa, sized [batch_size,].

        loss:
          (tensor) loss = SmoothL1Loss(fiiqa_preds, fiiqa_targets)
        '''
        #使用分类概率和估计值相乘再求和来求期望的方法,比直接分类和直接回归的效果更好。
        #先求得分类概率
        fiiqa_prob = F.softmax(fiiqa_preds,dim=1)
        #利用分类概率与对应预测值相乘后累加求和，求得期望值
        fiiqa_expect = torch.sum(Variable(torch.arange(0,200)).cuda().float()*fiiqa_prob, 1)
        #loss是期望值与ground trouth 之间的误差
        fiiqa_loss = F.smooth_l1_loss(fiiqa_expect, fiiqa_targets.float())
        return fiiqa_loss 
