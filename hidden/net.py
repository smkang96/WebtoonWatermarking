import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import Encoder, Decoder, Discriminator
from common.noise_layers import Noiser


def cover_labels(size):
    return torch.zeros(size)


def encoded_labels(size):
    return torch.ones(size)


class Hidden(nn.Module):
    def __init__(self, args, data):
        super().__init__()

        if args.mode in ['train', 'test']:
            self.enc_scale, self.dec_scale, self.adv_scale = args.enc_scale, args.dec_scale, args.adv_scale

        self.encoder = Encoder(data.w, data.h, data.l)
        self.noiser = Noiser(args.noise_type)
        self.decoder = Decoder(data.l)
        self.D = Discriminator()

        self.D_optim = torch.optim.Adam(self.D.parameters())
        self.G_optim = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))

    def to(self, device):
        self.device = device
        return super().to(device)

    def D_loss(self, img, encoded_img):
        encoded_img = encoded_img.detach()

        x = torch.cat([img, encoded_img])
        y = torch.cat([cover_labels(len(img)), encoded_labels(len(encoded_img))]).to(self.device)
        logits = self.D(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        return loss

    def G_loss(self, enc_loss, dec_loss, adv_loss):
        return self.enc_scale * enc_loss + self.dec_scale * dec_loss + self.adv_scale * adv_loss

    def enc_loss(self, encoded_img, img):
        return F.mse_loss(encoded_img, img)

    def dec_loss(self, decoded_msg, msg):
        return F.binary_cross_entropy(decoded_msg, msg)

    def adv_loss(self, encoded_img):
        y = cover_labels(len(encoded_img)).to(self.device)
        logits = self.D(encoded_img)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss

