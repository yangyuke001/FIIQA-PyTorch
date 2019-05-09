from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from net import AGNet
from loss import AGLoss
from datagen import ListDataset

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch AGNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_correct = 0 # best number of age_correct 
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.RandomCrop(150, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_test = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root='./data/trainingset/train-faces/', list_file='./data/trainingset/new_4people_train_standard.txt', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)

testset = ListDataset(root='./data/validationset/val-faces/', list_file='./data/validationset/new_4people_val_standard.txt', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Model
net = AGNet()
#net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_correct = checkpoint['correct']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = AGLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    age_correct = 0
    for batch_idx, (inputs, age_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        age_targets = Variable(age_targets.cuda().float())
        optimizer.zero_grad()
        age_preds = net(inputs)
        loss = criterion(age_preds.float(), age_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        age_correct_i = accuracy(age_preds, age_targets)
        age_correct += age_correct_i
        total += len(inputs)
        print('train_loss: %.3f | avg_loss: %.3f | age_prec: %.3f (%d/%d) |  [%d/%d]'  \
            % (loss.item(), train_loss/(batch_idx+1),      \
               200.*age_correct/total, age_correct, total,  \
               batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    total = 0
    age_correct = 0
    for batch_idx, (inputs, age_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda())
        age_targets = Variable(age_targets.cuda())

        age_preds = net(inputs)
        loss = criterion(age_preds, age_targets)

        test_loss += loss.item()
        age_correct_i = accuracy(age_preds, age_targets)
        age_correct += age_correct_i
        total += len(inputs)
        print('test_loss: %.3f | avg_loss: %.3f | age_prec: %.3f (%d/%d) | [%d/%d]' \
            % (loss.item(), test_loss/(batch_idx+1),      \
               200.*age_correct/total, age_correct, total,  \
               batch_idx+1, len(trainloader)))

    # Save checkpoint
    global best_correct
    if age_correct  > best_correct:
        print('Saving..')
        best_correct = age_correct 
        state = {
            'net': net.module.state_dict(),
            'correct': best_correct,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

def accuracy(age_preds, age_targets):
    '''Measure batch accuracy.'''
    AGE_TOLERANCE = 5
    age_prob = F.softmax(age_preds)
    age_expect = torch.sum(Variable(torch.arange(0,200)).cuda().float()*age_prob, 1)
    age_correct = ((age_expect-age_targets.float()).abs() < AGE_TOLERANCE).long().sum().cpu().item()

    return age_correct


for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)
