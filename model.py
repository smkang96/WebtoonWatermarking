iport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''Neural Network Architecture'''

class upscaleLayer(nn.Module):
    '''upscales by 2x'''
    def __init__(self, conv_dim, next_conv_dim, size):
        super(upscaleLayer, self).__init__()
        self.conv_dim = conv_dim
        self.next_conv_dim = next_conv_dim
        self.layer = nn.Conv2d(conv_dim, 4*next_conv_dim, 3, padding=1)
        if size != 1:
            self.base_maker = nn.ZeroPad2d(size/2)
        else:
            self.base_maker = nn.ZeroPad2d((1, 0, 1, 0))
    
    def forward(self, x):
        ncd = self.next_conv_dim
        base = 0*x # hack so that base will be variable
        base = self.base_maker(base)[:, :ncd]
        out = self.layer(x)
        base[:, :, ::2, ::2] = out[:, :ncd]
        base[:, :, ::2, 1::2] = out[:, ncd:2*ncd]
        base[:, :, 1::2, ::2] = out[:, 2*ncd:3*ncd]
        base[:, :, 1::2, 1::2] = out[:, 3*ncd:4*ncd]
        return base
    
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, dropout=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.InstanceNorm2d(c_out))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    '''utilizes image-wise average RGB value for image colorization'''
    def __init__(self, img_size = 256, conv_dim=64):
        '''color method not diverse here because we use GAN loss'''
        super(Generator, self).__init__()
        conv_layers = []
        upscale_layers = []
        prev_dim = 1 # (outline)
        curr_size = img_size
        curr_dim = conv_dim
        while curr_size != 4: # stops at 4
            if prev_dim % conv_dim != 0: # a slight hack
                conv_layers.append(conv(prev_dim, curr_dim, 4, bn=False))
                self.last_layer = upscaleLayer(2*curr_dim, 3, img_size/2) # last channel color
            else:
                conv_layers.append(conv(prev_dim, curr_dim, 4))
                upscale_layers.append(upscaleLayer(2*curr_dim, prev_dim, curr_size/2))
            prev_dim = curr_dim
            if curr_dim < conv_dim * 8:
                curr_dim *= 2
            curr_size /= 2
        
        conv_layers.append(conv(curr_dim, curr_dim, 3, 1, 0))
        conv_layers.append(conv(curr_dim, curr_dim, 2, 1, 0))
        upscale_layers.append(upscaleLayer(2*curr_dim, curr_dim, 2))
        upscale_layers.append(upscaleLayer(curr_dim, curr_dim, 1))
        
        self.down_net = nn.Sequential(*conv_layers)
        self.up_net = nn.Sequential(*reversed(upscale_layers))
        
    def forward(self, img):
        out = img
        outs = [] # used to feed to up-network later
        for layer in self.down_net:
            out = F.relu(layer(out))
            outs.append(out)
        for l_idx, layer in enumerate(self.up_net):
            out = F.relu(layer(out))
            out = torch.cat([outs[-(l_idx+2)], out], dim=1)
        out = F.tanh(self.last_layer(out))
        return out

class Decoder(nn.Module):
    """Decoder containing 5 convolutional layers."""
    def __init__(self, image_size=256, conv_dim=64, c_num=31):
        super(Discriminator, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False) # grayscale
        self.conv2 = conv(conv_dim, conv_dim*2, 4, bn=False)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, bn=False)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4, bn=False)
        self.conv5 = conv(conv_dim*4, conv_dim*4, 4, bn=False)
        self.linear = nn.Linear(128*7*7, 31)
        
    def forward(self, x):                          # If image_size is 256, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 128, 128) 
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 64, 64)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 32, 32)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 256, 16, 16)
        out = F.leaky_relu(self.conv5(out), 0.05)  # (?, 256, 8, 8)
        out = out.view(-1, 256*8*8)
        out = F.sigmoid(self.linear(out))
        return out
# g = Generator()
# print g(Variable(torch.randn(16, 1, 256, 256)))
