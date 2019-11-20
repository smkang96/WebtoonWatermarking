import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import ConvBNRelu
from common.noise_layers import Noiser


fc_dims = [32, 128, 512]
fc_conv_shape = (-1, 32, 4, 4)
conv_dims = [32] * 5 + [3]

pretrain_depth = 6
max_depth = len(fc_dims) + len(conv_dims)


class DFW(nn.Module):
    def __init__(self, args, data):
        super().__init__()

        self.l = data.l
        self.encoder = Encoder(args.img_size, data.l)
        self.noiser = Noiser(args.noise_type)
        self.decoder = Decoder(data.l)

        self.set_depth(max_depth)

    def set_depth(self, d):
        if d > max_depth:
            raise ValueError(f'Max depth is {max_depth}')
        self.encoder.depth = d
        self.decoder.depth = d


class PreReLU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(F.relu(x))

class PostReLU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return F.relu(self.module(x))

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Encoder(nn.Module):
    def __init__(self, img_size, l):
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(l, fc_dims[0]))
        prev_dim = fc_dims[0]
        for i in range(1, len(fc_dims)):
            self.layers.append(PreReLU(nn.Linear(prev_dim, fc_dims[i])))
            prev_dim = fc_dims[i]

        self.layers.append(View(fc_conv_shape))

        prev_dim = fc_conv_shape[1]
        for i in range(len(conv_dims)):
            self.layers.append(PreReLU(nn.ConvTranspose2d(prev_dim, conv_dims[i], kernel_size=2, stride=2)))
            prev_dim = conv_dims[i]

    def forward(self, msg):
        d = self.depth
        if self.depth > len(fc_dims):
            d += 1  # for View layer
        
        x = msg
        for i in range(d):
            x = self.layers[i](x)

        return x

class Decoder(nn.Module):
    def __init__(self, l):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(fc_dims[0], l))
        prev_dim = fc_dims[0]
        for i in range(1, len(fc_dims)):
            self.layers.append(PostReLU(nn.Linear(fc_dims[i], prev_dim)))
            prev_dim = fc_dims[i]

        self.layers.append(View((-1, fc_dims[-1])))

        prev_dim = fc_conv_shape[1]
        for i in range(len(conv_dims)):
            self.layers.append(PostReLU(nn.Conv2d(conv_dims[i], prev_dim, kernel_size=2, stride=2)))
            prev_dim = conv_dims[i]
 
    def forward(self, msg):
        d = self.depth
        if self.depth > len(fc_dims):
            d += 1  # for View layer
        
        x = msg
        for i in range(d-1, -1, -1):
            x = self.layers[i](x)

        return x

