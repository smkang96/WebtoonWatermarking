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
    def __init__(self, img_size, msg_l, train, dev):
        img_dir = os.environ['YUMI_DIR']
        self.w, self.h = img_size, img_size
        self.l = msg_l

        if train:
            img_dir += 'train/'
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        elif dev:
            img_dir += 'dev/'
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        else:
            img_dir += 'test/'
            self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        
        self.img_paths = list(filter(lambda x: 'jpg' in x, self.img_paths))

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        self.msg_dist = torch.distributions.Bernoulli(probs=0.5*torch.ones(msg_l))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = normalize(img)

        msg = self.msg_dist.sample()

        return img, msg

    def __len__(self):
        return len(self.img_paths)

