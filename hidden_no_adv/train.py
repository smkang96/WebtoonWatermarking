from multiprocessing import Process, Queue

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from data.watermark import Watermark
from net import Hidden
import common.path as path
from test import test_worker


log_filename = './train.log'


class HiddenTrain(Hidden):
    def optimize(self, img, msg):
        self.train()

        encoded_img = self.encoder(img, msg)
        noised_img = self.noiser(encoded_img)
        decoded_msg = self.decoder(noised_img)
        
        stats = self.optimize_G(img, msg, encoded_img, decoded_msg)

        return stats

    def optimize_G(self, img, msg, encoded_img, decoded_msg):
        self.G_optim.zero_grad()

        enc_loss = self.enc_loss(encoded_img, img)
        dec_loss = self.dec_loss(decoded_msg, msg)
        G_loss = self.G_loss(enc_loss, dec_loss)
        G_loss.backward()
        self.G_optim.step()

        return {
            'G_loss': G_loss.item(),
            'enc_loss': enc_loss.item(),
            'dec_loss': dec_loss.item(),
        }


def start_test_process(args):
    queue = Queue()
    test_process = Process(target=test_worker, args=(args, queue))
    test_process.start()

    return test_process, queue


def train(args):
    test_process, queue = start_test_process(args)
    log_file = open(log_filename, 'w+', buffering=1)

    dataset = Watermark(args.img_size, args.msg_l, train=True)
    net = HiddenTrain(args, dataset).to(args.device)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    with trange(args.epochs, unit='epoch') as tqdm_bar:
        for epoch_i in tqdm_bar:
            for batch_i, (img, msg) in enumerate(loader):
                img, msg = img.to(args.device), msg.to(args.device)
                stats = net.optimize(img, msg)
                tqdm_bar.set_postfix(**stats)

            if epoch_i % args.save_freq == 0:
                log_file.write("Epoch {} | {}\n".format(epoch_i,  " ".join([f"{k}: {v:.3f}" for k, v in stats.items()])))
                torch.save(net.state_dict(), path.save_path)

            if epoch_i % args.test_freq == 0:
                queue.put((epoch_i, net.state_dict()))

    log_file.close()
    queue.join()
    test_process.kill()

