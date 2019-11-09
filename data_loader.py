#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.
    
    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """
    def __init__(self, root, train, transform=None):
        """Initializes image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        if train:
            self.image_paths = list(filter(lambda x: 'test_' not in x, self.image_paths))
        else:
            self.image_paths = list(filter(lambda x: 'test_' in x, self.image_paths))
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(list(self.image_paths))

    
def get_dir_loader(image_path, image_size, batch_size, train=True, num_workers=2):
    """Builds and returns Dataloader."""
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor()
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ImageFolder(image_path, train, transform=transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
