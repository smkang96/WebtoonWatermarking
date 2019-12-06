import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import ConvBNRelu
from common.noise_layers import Noiser


encoder_fc_dims = [32, 128, 512]
fc_conv_shape = (-1, 32, 4, 4)
conv_dims = [32] * 5 + [3]

avg_pool_shape = (3, 3)
decoder_fc_dims = [32, 128, avg_pool_shape[0]*avg_pool_shape[1]*fc_conv_shape[1]]

pretrain_depth = 6
max_depth = len(encoder_fc_dims) + len(conv_dims)


class DFW(nn.Module):
    def __init__(self, args, data):
        super().__init__()

        self.l = args.msg_l
        self.encoder = Encoder(args.img_size, 31)
        self.noiser = Noiser(args.noise_type)
        self.decoder = Decoder(31)

        self.set_depth(max_depth)

    def set_depth(self, d):
        if d > max_depth:
            raise ValueError(f'Max depth is {max_depth}')
        self.encoder.depth = d
        self.decoder.depth = d

    def to(self, device):
        self.noiser.to(device)
        return super().to(device)


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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)


class Encoder(nn.Module):
    def __init__(self, img_size, l):
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(l, encoder_fc_dims[0]))
        prev_dim = encoder_fc_dims[0]
        for i in range(1, len(encoder_fc_dims)):
            self.layers.append(PreReLU(nn.Linear(prev_dim, encoder_fc_dims[i])))
            prev_dim = encoder_fc_dims[i]

        self.layers.append(View(fc_conv_shape))

        prev_dim = fc_conv_shape[1]
        for i in range(len(conv_dims)):
            self.layers.append(PreReLU(nn.ConvTranspose2d(prev_dim, conv_dims[i], kernel_size=2, stride=2)))
            prev_dim = conv_dims[i]

    def forward(self, msg):
        d = self.depth
        if self.depth > len(encoder_fc_dims):
            d += 1  # for View layer
        
        x = msg
        for i in range(d):
            x = self.layers[i](x)

        return x

class Decoder(nn.Module):
    def __init__(self, l):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(decoder_fc_dims[0], l))
        prev_dim = decoder_fc_dims[0]
        for i in range(1, len(decoder_fc_dims)):
            self.layers.append(PostReLU(nn.Linear(decoder_fc_dims[i], prev_dim)))
            prev_dim = decoder_fc_dims[i]

        self.layers.append(Flatten())
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=avg_pool_shape))

        prev_dim = fc_conv_shape[1]
        for i in range(len(conv_dims)):
            self.layers.append(PostReLU(nn.Conv2d(conv_dims[i], prev_dim, kernel_size=2, stride=2)))
            prev_dim = conv_dims[i]
 
    def forward(self, msg):
        d = self.depth
        if self.depth > len(decoder_fc_dims):
            d += 2  # for avg pool & flatten layer
        
        x = msg
        for i in range(d-1, -1, -1):
            x = self.layers[i](x)

        return x

