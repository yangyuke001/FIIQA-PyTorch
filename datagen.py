from __future__ import print_function

import io
import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.transform = transform

        self.fname = []
        self.age = []

        with io.open(list_file, encoding='gbk') as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            sp = line.strip().split()
            self.fname.append(sp[0])
            self.age.append(int(sp[1]))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          age: (float) age.
        '''
        # Load image and bbox locations.
        fname = self.fname[idx]
        age = self.age[idx]

        img = Image.open(os.path.join(self.root, fname))
        img = self.transform(img)
        return img, age

    def __len__(self):
        return self.num_imgs


def test():
    import torchvision
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='./data/validationset/val-faces/', list_file='./data/validationset/val-faces/new_4people_val_standard.txt', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    for img, age in dataloader:
        print(img.size())
        print(age.size())

