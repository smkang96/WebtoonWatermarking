import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.backends import cudnn
cudnn.benchmark = True

import numpy as np
from random import randint

import model
import data_loader

def to_var(x, volatile=False):
    x = x.cuda()
    return Variable(x, volatile=volatile)

def normalize_img(img):
    return img * 2 - 1

def denorm_img(img):
    return (img + 1) / 2. 

img_size = (256, 256)
pix_num = img_size[0] * img_size[1]
batch_size = 32
epoch_num = 200
img_path = '../yumi_data/yumi/_target/' # to model natural cartoon img
test_path = './samples/'
model_dir = './model/'
model_name = '%d_yumi_%s.pth'
model_path = model_dir + model_name
sample_path = './samples/'
conv_dim = 64

enc = model.Encoder(conv_dim=conv_dim)
enc.cuda()
enc_optimizer = torch.optim.Adam(enc.parameters(), lr = 6e-4)
dec = model.Decoder(conv_dim=conv_dim)
dec.cuda()
dec_optimizer = torch.optim.Adam(dec.parameters(), lr = 2e-4)
data_iterer = data_loader.get_dir_loader(img_path, img_size[0], batch_size) # TODO: test
data_num = len(data_iterer)
bern_dist = torch.distributions.bernoulli.Bernoulli(probs=0.5*torch.ones(batch_size, 31))

print('preparation ready. training...')

for e_idx in range(epoch_num):
    for b_idx, img in enumerate(data_iterer):
        # Normalize
        norm_img = normalize_img(img).cuda()
        
        # get user encoding
        uids = bern_dist.sample().cuda()
        enc_uids = uids[:norm_img.size(0)] # prevent errors from unholy data sizes
        
        # enc training #
        watermarks = enc(enc_uids.view(-1, 31, 1, 1))
        g_loss_l2 = torch.mean(watermarks**2)
        new_img = norm_img + watermarks + 0.2*torch.randn(watermarks.size()).cuda()
        recov_enc_uids = dec(new_img)
        ce_loss = -(enc_uids * (recov_enc_uids + 1e-5).log() + 
                    (1-enc_uids) * ((1-recov_enc_uids) + 1e-5).log())
        g_loss_d = torch.mean(ce_loss)
        g_loss = g_loss_l2 + g_loss_d
        enc_optimizer.zero_grad()
        g_loss.backward()
        enc_optimizer.step()

        # dec training #
        watermarks = enc(enc_uids.view(-1, 31, 1, 1))
        new_img = norm_img + watermarks + 0.2*torch.randn(watermarks.size()).cuda()
        recov_enc_uids = dec(new_img)
        ce_loss = -(enc_uids * (recov_enc_uids + 1e-5).log() + 
                    (1-enc_uids) * ((1-recov_enc_uids) + 1e-5).log())
        d_loss = torch.mean(ce_loss)
        dec_optimizer.zero_grad()
        d_loss.backward()
        dec_optimizer.step()
                
        # logging to console
        if (b_idx+1) % 100 == 0:
            new_uids = bern_dist.sample().cuda()
            enc_uids = uids
            watermarks = enc(enc_uids.view(-1, 31, 1, 1))
            recov_enc_uids = dec(watermarks)
            bin_recov_enc_uids = (recov_enc_uids > 0.5).float()
            acc = torch.sum((bin_recov_enc_uids == enc_uids).float()) / (enc_uids.size(0) * enc_uids.size(1))
            diff = torch.mean(torch.abs(recov_enc_uids[:-1] - recov_enc_uids[1:]))
            print("[{}|{}/{}] "
                  "g_loss_l2: {:.4f}, g_loss_d: {:.4f}, acc: {:.1f}%, diff: {:.4f}". \
                  format(e_idx+1, b_idx+1, data_num+1, g_loss_l2.item(), g_loss_d.item(),
                         acc*100, diff))
    
    # logging images
    new_uids = bern_dist.sample().cuda()
    enc_uids = uids
    watermarks = enc(enc_uids.view(-1, 31, 1, 1))
    save_image(denorm_img(watermarks.data[:16]),
               sample_path + '{}_val.png'.format(e_idx+1), 
               nrow=4)
    
    if e_idx % 50 == 49:
        enc_path_name = model_path % (e_idx+1, 'enc')
        torch.save(enc.state_dict(), enc_path_name)
        dec_path_name = model_path % (e_idx+1, 'dec')
        torch.save(dec.state_dict(), dec_path_name)
        print(f'Models saved')