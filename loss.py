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
        fiiqa_prob = F.softmax(fiiqa_preds,dim=1)
        #print('fiiqa_prob: %.3f ' % (fiiqa_prob))
        fiiqa_expect = torch.sum(Variable(torch.arange(0,200)).cuda().float()*fiiqa_prob, 1) #.float()
        fiiqa_loss = F.smooth_l1_loss(fiiqa_expect, fiiqa_targets.float())#.float()
        #print('fiiqa_loss: %.3f ' % fiiqa_loss.data[0])
        return fiiqa_loss 
