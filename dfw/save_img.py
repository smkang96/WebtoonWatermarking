import os

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.watermark import Watermark, denormalize
from net import DFW, max_depth
import common.path as path
from utils import HammingCoder
    
def store_images(img, msg, watermark, encoded_img, noised_img, decoded_msg, save_dir):
    convert = lambda img: np.moveaxis(denormalize(img).cpu().numpy(), [1, 2, 3], [3, 1, 2])
    img = convert(img)
    watermark = convert(watermark / abs(watermark).max())
    encoded_img = convert(encoded_img)
    noised_img = convert(noised_img)
    msg = msg.cpu().numpy()
    decoded_msg = (decoded_msg>0.5).float().cpu().numpy()

    dict_output_info = {}
    for i in range(img.shape[0]):
        fig = plt.figure()
        gridspec = fig.add_gridspec(ncols=6, nrows=1, width_ratios=[2, 2, 2, 2, 1, 1])
        axes = [fig.add_subplot(gridspec[0, i]) for i in range(6)]
        ax1, ax2, ax3, ax4, ax5, ax6 = axes

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.set_title('Original\nImage')
        ax1.imshow(img[i])

        ax2.set_title('Watermark\n(Normalized)')
        ax2.imshow(watermark[i])

        ax3.set_title('Encoded\nImage')
        ax3.imshow(encoded_img[i])

        ax4.set_title('Noised\nImage')
        ax4.imshow(noised_img[i])

        ax5.set_title('Original\nMsg')
        ax5.imshow(msg[i][:, None], cmap='gray', aspect=2/31)

        ax6.set_title('Decoded\nMsg')
        ax6.imshow(decoded_msg[i][:, None], cmap='gray', aspect=2/31)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{i}.png'), bbox_inches='tight')
        plt.close()
        
        fig2 = plt.figure()
        plt.imshow(noised_img[i])
        fig2.savefig(os.path.join(save_dir, f'{i}noise.png'))
        plt.close()
        
        fig3 = plt.figure()
        plt.imshow(img[i])
        fig3.savefig(os.path.join(save_dir, f'{i}original.png'))
        plt.close()
        
        dict_output_info[i] = sum(abs(decoded_msg[i]-msg[i])) #number of errors
    print(dict_output_info)
    
def save_img(args):
    dataset = Watermark(args.img_size, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.n_imgs, shuffle=False)
    msg_dist = torch.distributions.Bernoulli(probs=0.5*torch.ones(args.msg_l))
    
    net = DFW(args, dataset).to(args.device)
    net.set_depth(max_depth)
    net.load_state_dict(torch.load(path.save_path))
    net.eval()

    hamming_coder = HammingCoder(device=args.device)
    
    save_dir = './examples_' + args.noise_type
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    with torch.no_grad():
        img = next(iter(loader))
        msg = msg_dist.sample([img.shape[0]])
        img, msg = img.to(args.device), msg.to(args.device)
        
        hamming_msg = torch.stack([hamming_coder.encode(x) for x in msg])
        watermark = net.encoder(hamming_msg)
        encoded_img = (img + watermark).clamp(-1, 1)
        noised_img, _ = net.noiser([encoded_img, img])
        decoded_msg_logit = net.decoder(noised_img)

        pred_without_hamming_dec = (torch.sigmoid(decoded_msg_logit) > 0.5).int()
        pred_msg = torch.stack([hamming_coder.decode(x) for x in pred_without_hamming_dec])
        
        store_images(img, msg, watermark, encoded_img, noised_img, pred_msg, save_dir)
            
