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
    return img / 128. - 1

def denorm_img(img):
    return (img + 1) / 2. 

img_size = (256, 256)
pix_num = img_size[0] * img_size[1]
batch_size = 16
epoch_num = 200
img_path = '../../NewColorer/yumi_train/' # to model natural cartoon img
test_path = '../../NewColorer/yumi_test/'
model_dir = './model/'
model_name = '%d_yumi_global_gen.pth'
model_path = model_dir + model_name
sample_path = './yumi_up/'
log_path = './log_up/'
conv_dim = 64

G = model.Generator(conv_dim=conv_dim)
G.cuda()
G_optimizer = torch.optim.Adam(G.parameters(), lr = 6e-4)
data_iterer = data_loader.get_dir_loader(img_path, img_size[0], batch_size)
data_num = len(data_iterer)

# below: just to get cartoon test images
for _ in xrange(1):
    for img, _ in data_loader.get_dir_loader(test_path, img_size[0], batch_size):
        # get the data
        img = img * 255
        img = to_var(img)
        outline = torch.mean(img[:, :, :, 256:], dim=1, keepdim=True)
        color_img = img[:, :, :, 256:]
        # Normalize
        norm_color_img = normalize_img(color_img)
        norm_outline = normalize_img(outline)
        test_G_input = norm_outline
        test_G_img = norm_color_img
        save_image(denorm_img(test_G_img.data),
                   sample_path + 'ground_truth.png', 
                   nrow=4)
        break

max_crop = 30
        
print 'preparation ready. training...'

for e_idx in xrange(epoch_num):
    for b_idx, (img, _) in enumerate(data_iterer):
        # get the data
        img = img * 255
        img = to_var(img)
        outline = torch.mean(img[:, :, :, 256:], dim=1, keepdim=True) # actually gray
        color_img = img[:, :, :, 256:]
        
        ## data augmentation: random crop
        hor_lbound = randint(0, max_crop)
        hor_rbound = randint(0, max_crop)
        ver_tbound = randint(0, max_crop)
        ver_bbound = randint(0, max_crop)
        if hor_rbound != 0:
            outline = outline[:, :, hor_lbound:-hor_rbound]
            color_img = color_img[:, :, hor_lbound:-hor_rbound]
        else:
            outline = outline[:, :, hor_lbound:]
            color_img = color_img[:, :, hor_lbound:]
        if ver_bbound != 0:
            outline = outline[:, :, :, ver_tbound:-ver_bbound]
            color_img = color_img[:, :, :, ver_tbound:-ver_bbound]
        else:
            outline = outline[:, :, :, ver_tbound:]
            color_img = color_img[:, :, :, ver_tbound:]
            
        outline = nn.Upsample(size=img_size, mode='bilinear')(outline)
        color_img = nn.Upsample(size=img_size, mode='bilinear')(color_img)
        ##################################
        
        # Normalize
        norm_color_img = normalize_img(color_img)
        norm_outline = normalize_img(outline)
        
        # G training #
        fake_img = G(norm_outline)
        g_loss_l1 = torch.mean(torch.abs(norm_color_img - fake_img))
        g_loss = 10. * g_loss_l1
        G_optimizer.zero_grad()
        g_loss.backward()
        G_optimizer.step()
                
        # logging to console
        if (b_idx+1) % 100 == 0:
            print("[{}|{}/{}] "
                  "g_loss_rec: {:.4f}". \
                  format(e_idx+1, b_idx+1, data_num+1, g_loss_l1.data[0]))
    
    # logging images
    val_img = G(test_G_input)
    save_image(denorm_img(val_img.data),
               sample_path + '{}_val.png'.format(e_idx+1), 
               nrow=4)
    save_image(denorm_img(fake_img.data),
               sample_path + '{}_train.png'.format(e_idx+1), 
               nrow=4)
    
    if e_idx % 50 == 9:
        new_path_name = model_path % (e_idx+1,)
        torch.save(G.state_dict(), new_path_name)
        print 'Model saved to', new_path_name