import numpy as np
import torch.nn as nn
from . import Identity, Crop, Cropout, Dropout, Resize, JpegCompression, Quantization
import random
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)

class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_type, amount_noise=0):
        super(Noiser, self).__init__()

        self.noise_layers = nn.ModuleList()
        self.noise_layers.append(Identity())
        self.noise_type = noise_type
        self.amount_noise = amount_noise

        if noise_type == 'no_noise':
            pass
        elif noise_type == 'combined':
            self.noise_layers.append((Crop((0.8, 0.9),(0.8, 0.9))))
            self.noise_layers.append(Cropout((0.8, 0.8),(0.8, 0.9)))
            self.noise_layers.append(Dropout((0.25, 0.35)))
            self.noise_layers.append(Resize((0.8, 0.6)))
            self.noise_layers.append(JpegCompression())
        elif noise_type == 'crop':
            self.noise_layers.append(Crop((0.8, 0.9),(0.8, 0.9)))
        elif noise_type == 'cropout':
            self.noise_layers.append(Cropout((0.8, 0.8),(0.8, 0.9)))
        elif noise_type == 'dropout':
            self.noise_layers.append(Dropout((0.25, 0.35)))
        elif noise_type == 'resize':
            self.noise_layers.append(Resize((0.8, 0.6)))
        elif noise_type == 'jpeg':
            self.noise_layers.append(JpegCompression())
        else:
            raise NotImplementedError 

    def to(self, device):
        for module in self.noise_layers:
            if type(module) == JpegCompression:
                module.to(device)
        return super().to(device)

    def forward(self, encoded_and_cover):
        if self.noise_type == "combined":
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            if random.random() <= self.amount_noise:
                return random_noise_layer(encoded_and_cover)
            else:
                return encoded_and_cover
        else:
            encoded_and_cover = self.noise_layers[0](encoded_and_cover)
            for layer in self.noise_layers:
                encoded_and_cover = layer(encoded_and_cover)
            return encoded_and_cover



