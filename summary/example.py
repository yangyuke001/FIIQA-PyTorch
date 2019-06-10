import torch
from misc.summary import model_summary
from model.MobileNetV2 import MobileNetV2


if __name__ == '__main__':
    net = MobileNetV2()
    model_summary(net, input_size=(3, 224, 224))