'''Train CK+ with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms,utils
from torch.utils.data import DataLoader
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from CK import CK
from torch.autograd import Variable
from models import *
from ShuffleNetV2 import ShuffleNetV2
from flops_counter_pytorch.ptflops import get_model_complexity_info
from summary import model_summary
from datagen import ListDataset


train_data = '../train_val_imgs/Manually/Manually_train_croped'
test_data = '../train_val_imgs/Manually/Manually_validation_croped'

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
#parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 1 # 5
learning_rate_decay_rate = 0.8 # 0.9


total_epoch = 500
bs = 128
input_size = 64
cut_size = input_size - 1
n_class=7

#path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))
path = './AffectNet+ShuffleNetV2/'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(input_size),
    #transforms.RandomCrop(cut_size),
    #transforms.RandomHorizontalFlip(),
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #transforms.ToTensor(),
    #transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

#trainset=torchvision.datasets.ImageFolder(train_data,transform_train)
trainset = ListDataset(root='../train_val_imgs/Manually/Manually_train_croped/', list_file='./AffectNet/train.txt', transform=transform_train)
trainloader=DataLoader(trainset,bs,shuffle=True, num_workers=12)
#testset=torchvision.datasets.ImageFolder(test_data,transform_test)
testset = ListDataset(root='../train_val_imgs/Manually/Manually_validation_croped/', list_file='./AffectNet/val.txt', transform=transform_test)
testloader=DataLoader(testset,batch_size=128,shuffle=True, num_workers=12)


net = ShuffleNetV2(input_size,n_class)
'''
model_summary(net,input_size=(3,input_size,input_size))
flops, params = get_model_complexity_info(net, (input_size, input_size), as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops)
print('Params: ' + params)
#net = net.to(device=my_device)
'''

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'ShuffleNetV2.pth'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer=optim.AdaBound(net.parameters(), lr=opt.lr, final_lr=0.1)
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        #inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1) #clip_gradient能有效了控制梯度爆炸的影响，使得最终的loss能下降到满意的结果 
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total

# test
def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), \
            100. * correct / total, correct, total))
    # Save checkpoint.
    Test_acc = 100.*correct/total

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.dataset + '_' + 'ShuffleNetV2'):
            os.mkdir(opt.dataset + '_' + 'ShuffleNetV2')
        if not os.path.isdir(path):
            os.mkdir(path)
        #torch.save(state, os.path.join(path, str(best_Test_acc) + '_ShuffleNetV2.pth'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch
        torch.save(state, os.path.join(path, str(best_Test_acc) + '_'+str(input_size)+'_ShuffleNetV2.pth'))

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    test(epoch)

print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)