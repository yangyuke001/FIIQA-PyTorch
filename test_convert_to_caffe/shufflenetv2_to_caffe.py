import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import caffe
import numpy as np

this_dir = '.'

cpath = '/home/jinsy/2T/jinsy/code/tuya_sdk/Pytorch2Caffe'
if cpath not in sys.path:
    sys.path.append(cpath)

import converter
from utils.load_test_image import load_test_image
from utils.get_topk import get_topk
from utils.load import load_troch_model2
import shufflenetv2 as ShuffleNetV2


def _channel_shuffle(raw, input, groups):
    x = input.clone()
    name = converter.log.add_layer(name='shuffle_channel')
    converter.log.add_blobs([x], name='shuffle_channel')
    layer = converter.caffe_net.Layer_param(name=name, type='ShuffleChannel',
                                            bottom=[converter.log.blobs(input)], top=[converter.log.blobs(x)])
    layer.channel_shuffle_param(groups)
    converter.log.cnet.add_layer(layer)
    return x


ShuffleNetV2.channel_shuffle = converter.Rp(ShuffleNetV2.channel_shuffle, _channel_shuffle)


def parse():
    parser = argparse.ArgumentParser(description='convert pytorch model to caffe')
    parser.add_argument('-c', '--convert', action='store_true', help='convert pytorch model to caffe')
    parser.add_argument('-t', '--test', action='store_true', help='test converted model')
    parser.add_argument('-wm', '--width-mult', type=float, choices=[0.5, 1.0], default=1.0,
                        help='width mult')
    return parser.parse_args()


def convert(pytorch_net, caffe_prototxt, caffe_model_file, name):
    pytorch_net.eval()
    input = torch.ones(1, 3, 160, 160)
    converter.trans_net(pytorch_net, input, name)
    converter.save_prototxt(caffe_prototxt)
    converter.save_caffemodel(caffe_model_file)


def test(caffe_prototxt, caffe_model_file, pytorch_net, topk=5):
    pytorch_net.eval()
    input = load_test_image()
    with torch.no_grad():
        x1 = pytorch_net(input)
        x1 = x1.numpy()

    caffe_net = caffe.Net(caffe_prototxt, caffe_model_file, caffe.TEST)
    caffe.set_mode_cpu()
    caffe_net.blobs['data1'].data.reshape(*input.size())
    caffe_net.blobs['data1'].data[...] = input.numpy()
    x2 = caffe_net.forward()['softmax1']
    idx = 0

    expect = torch.dot(torch.arange(0,200).float() ,torch.from_numpy(x1.flatten()))
    print('pytorch: ', expect)

    expect = torch.dot(torch.arange(0,200).float() ,torch.from_numpy(x2.flatten()))
    print('caffe: ', expect)


class SoftmaxWrapper(nn.Module):
    def __init__(self, net):
        super(SoftmaxWrapper, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return F.softmax(y, dim=1)


def shufflenetv2(width_mult=1., pretrained=True):
    model = ShuffleNetV2.ShuffleNetV2(input_size=160, width_mult=width_mult)
    if pretrained:
        if width_mult == 1.:
            model_file = osp.join(this_dir, 'model/97_160_2.pth')
        elif width_mult == .5:
            model_file = osp.join(this_dir, '../models/ShuffleNetV2/shufflenetv2_x0.5_60.646_81.696.pth.tar')
        else:
            raise ValueError('width_mult = {} is not support pretrained model'.format(width_mult))
        assert osp.exists(model_file), '{} is not exists!'.format(model_file)
        print('shufflenetv2 load state dict: ', model_file)
        state = torch.load(model_file)
        model.load_state_dict(state['net'], strict=False)

    return model


if __name__ == '__main__':
    args = parse()

    name = 'shufflenetv2_{:.1f}x'.format(args.width_mult)

    caffe_prototxt = osp.join(this_dir, 'model/') + '{}.prototxt'.format(name)
    caffe_model_file = osp.join(this_dir, 'model/') + '{}.caffemodel'.format(name)
    net = shufflenetv2(width_mult=args.width_mult, pretrained=True)
    net = SoftmaxWrapper(net)

    if args.convert:
        convert(net, caffe_prototxt, caffe_model_file, name)

    if args.test:
        test(caffe_prototxt, caffe_model_file, net)
