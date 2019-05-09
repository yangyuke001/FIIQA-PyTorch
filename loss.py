from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class AGLoss(nn.Module):
    def __init__(self):
        super(AGLoss, self).__init__()

    def forward(self, age_preds, age_targets):
        '''Compute loss (age_preds, age_targets) .

        Args:
          age_preds: (tensor) predicted ages, sized [batch_size,100].
          age_targets: (tensor) target ages, sized [batch_size,].

        loss:
          (tensor) loss = SmoothL1Loss(age_preds, age_targets)
        '''
        age_prob = F.softmax(age_preds)
        #print('age_prob: %.3f ' % (age_prob))
        age_expect = torch.sum(Variable(torch.range(0,199)).cuda().float()*age_prob, 1)
        age_loss = F.smooth_l1_loss(age_expect, age_targets.float())
        #print('age_loss: %.3f ' % age_loss.data[0])
        return age_loss 
