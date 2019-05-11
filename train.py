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
from shufflenetv2 import shufflenetv2



parser = argparse.ArgumentParser(description='PyTorch AGNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_correct = 0 # best number of fiiqa_correct 
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    #transforms.RandomCrop(224, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root='./data/trainingset/train-faces/', list_file='./data/trainingset/new_4people_train_standard.txt', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testset = ListDataset(root='./data/validationset/val-faces/', list_file='./data/validationset/new_4people_val_standard.txt', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Model
net = AGNet()
#net = shufflenetv2()
#net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_correct = checkpoint['correct']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) #多gpu并行训练
net.cuda()

criterion = AGLoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.AdaBound(net.parameters(),lr=args.lr,final_lr=0.1)

#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    fiiqa_correct = 0
    for batch_idx, (inputs, fiiqa_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        fiiqa_targets = Variable(fiiqa_targets.cuda()) 
        optimizer.zero_grad()
        fiiqa_preds = net(inputs)
        loss = criterion(fiiqa_preds.float(), fiiqa_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        fiiqa_correct_i = accuracy(fiiqa_preds, fiiqa_targets)
        fiiqa_correct += fiiqa_correct_i
        total += len(inputs)
        print('train_loss: %.3f | fiiqa_prec: %.3f (%d/%d) |  [%d/%d]'  \
            % (loss.item(),       \
               100.*fiiqa_correct/total, fiiqa_correct, total,  \
               batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    total = 0
    fiiqa_correct = 0
    for batch_idx, (inputs, fiiqa_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda())
        fiiqa_targets = Variable(fiiqa_targets.cuda())

        fiiqa_preds = net(inputs)
        loss = criterion(fiiqa_preds, fiiqa_targets)

        test_loss += loss.item()
        fiiqa_correct_i = accuracy(fiiqa_preds, fiiqa_targets)
        fiiqa_correct += fiiqa_correct_i
        total += len(inputs)
        print('test_loss: %.3f | fiiqa_prec: %.3f (%d/%d) | [%d/%d]' \
            % (loss.item(),       \
               100.*fiiqa_correct/total, fiiqa_correct, total,  \
               batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_correct
    if fiiqa_correct  > best_correct:
        print('Saving..')
        best_correct = fiiqa_correct 
        state = {
            'net': net.module.state_dict(),
            'correct': best_correct,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

def accuracy(fiiqa_preds, fiiqa_targets):
    '''Measure batch accuracy.'''
    fiiqa_TOLERANCE = 5
    fiiqa_prob = F.softmax(fiiqa_preds)
    fiiqa_expect = torch.sum(Variable(torch.arange(0,200)).cuda().float()*fiiqa_prob, 1)
    fiiqa_correct = ((fiiqa_expect-fiiqa_targets.float()).abs() < fiiqa_TOLERANCE).long().sum().cpu().item()

    return fiiqa_correct


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
