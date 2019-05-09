'''Init AGNet18 with pretrained ResNet18 model.'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from net import AGNet, ResNet18


print('Loading pretrained ResNet18 model..')
d = torch.load('./model/resnet18.pth')

print('Loading into backbone..')
backbone = ResNet18()
dd = backbone.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving AGNet..')
net = AGNet()
net.backbone.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
print('Done!')
