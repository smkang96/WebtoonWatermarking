import torch.nn as nn
from .conv_bn_relu import ConvBNRelu


blocks = 3
channels = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(blocks-1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X).squeeze()
        return X
