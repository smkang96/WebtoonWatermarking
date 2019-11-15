import torch
import torch.nn as nn
from .conv_bn_relu import ConvBNRelu


blocks = 4
channels = 64


class Encoder(nn.Module):
    def __init__(self, w, h, l):
        super(Encoder, self).__init__()
        self.w, self.h = w, h
        self.conv_channels = channels
        self.num_blocks = blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + l,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.h, self.w)
        encoded_image = self.conv_layers(image)

        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        im_w = torch.tanh(im_w)
        return im_w
