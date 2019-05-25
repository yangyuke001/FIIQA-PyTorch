import os.path as osp
import cv2
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

this_dir = osp.abspath(osp.dirname(__file__))


def load_test_image(filename='test_face.jpg', image_size=(3, 160, 160)):
    full_filename = osp.join(this_dir, '../image/crop', filename)
    print(full_filename)
    rgb_image = Image.open(full_filename)
    tf = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])
    tensor = tf(rgb_image)
    return tensor.view(1, *image_size)


def load_test_image2(filename='cat_224x224.jpg', image_size=(3, 224, 224)):
    full_filename = osp.join(this_dir, '../data', filename)
    bgr_image = cv2.imread(full_filename)
    bgr_image2 = torch.from_numpy(cv2.resize(bgr_image, image_size[1:]).astype(np.float32))
    return bgr_image2.view(1, *image_size)
