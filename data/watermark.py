import torch
import torch.distributions
from torchvision import transforms
from torch.utils import data
import os
from PIL import Image

def normalize(img):
    return img * 2 - 1


def denormalize(img):
    return (img + 1) / 2


class Watermark(data.Dataset):
    def __init__(self, img_size, train, dev):
        img_dir = os.environ['YUMI_DIR']
        self.w, self.h = img_size, img_size

        if train:
            img_dir = os.path.join(img_dir, 'train')
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        elif dev:
            img_dir = os.path.join(img_dir, 'dev')
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        else:
            img_dir = os.path.join(img_dir, 'test')
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        
        self.img_paths = list(filter(lambda x: 'jpg' in x, self.img_paths))

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        img = normalize(img) 
        
        return img

    def __len__(self):
        return len(self.img_paths)

