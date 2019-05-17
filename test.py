import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F
from shufflenetv2 import ShuffleNetV2
import scipy.io as scio
import cv2

#config
matFile = 'pScores.mat'
fiiqaWeight = './model/97_160_2.pth'
detectFace = './model/haarcascade_frontalface_default.xml'
imagePath = './image/test.jpg'
facePath = './image/crop/test_face.jpg'
inputSize = 160

#crop face from img
faceCascade = cv2.CascadeClassifier(detectFace)
image = cv2.imread(imagePath)
faces = faceCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(10, 10)
)
for (x, y, w, h) in faces:
    cv2.imwrite('./image/crop/' + 'test_face.jpg',image[y:y+h,x:x+w])
cv2.waitKey(0)

#transfer img to tensor
dataTransforms =  transforms.Compose([
               transforms.Resize(inputSize),
               transforms.ToTensor(),
               transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

#load net 
net = ShuffleNetV2(inputSize)
checkpoint = torch.load(fiiqaWeight)
net.load_state_dict(checkpoint['net'])
net.eval()

#load face and get expect num
face = Image.open(facePath)
imgblob = dataTransforms(face).unsqueeze(0)
imgblob = Variable(imgblob)
torch.no_grad()
predict = F.softmax(net(imgblob),dim=1)
expect = torch.sum(Variable(torch.arange(0,200)).float()*predict, 1)
expect = int(expect)

#load matFile and get score
data = scio.loadmat(matFile)
scores = data['pScores']
score = scores[:,expect]

print('expect: %d' % expect)
print('score: %.3f' %  score)
