import numpy as np
import torch
from torch.utils.data import DataLoader

from data.watermark import Watermark
from net import Hidden
import path


log_filename = './test.log'


class HiddenTest(Hidden):
    def stats(self, img, msg):
        self.eval()

        encoded_img = self.encoder(img, msg)
        noised_img = self.noiser(encoded_img)
        decoded_msg = self.decoder(noised_img)

        enc_loss = self.enc_loss(encoded_img, img)
        dec_loss = self.dec_loss(decoded_msg, msg)
        G_loss = self.G_loss(enc_loss, dec_loss)

        return {
            'G_loss': G_loss,
            'enc_loss': enc_loss,
            'dec_loss': dec_loss,
        }


def test_worker(args, queue):
    log_file = open(log_filename, 'w+', buffering=1)

    dataset = Watermark(args.img_dir, args.img_size, args.msg_l, train=False)
    net = HiddenTest(args, dataset).to(args.device)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    
    while True:
        epoch_i, state_dict = queue.get()
        net.load_state_dict(state_dict)

        stats = {
            'G_loss': 0,
            'enc_loss': 0,
            'dec_loss': 0,
        }

        with torch.no_grad():
            for img, msg in loader:
                img, msg = img.to(args.device), msg.to(args.device)
                batch_stats = net.stats(img, msg)
                for k in stats:
                    stats[k] += len(img) * batch_stats[k]

        for k in stats:
            stats[k] = stats[k] / len(dataset)

        log_file.write("Epoch {} | {}\n".format(epoch_i,  " ".join([f"{k}: {v:.3f}"for k, v in stats.items()])))
        queue.task_done()

def test(args):
    dataset = Watermark(args.img_dir, args.img_size, args.msg_l, train=False)
    net = HiddenTest(args, dataset).to(args.device)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    
    net.load_state_dict(torch.load(path.save_path))

    stats = {
        'G_loss': 0,
        'enc_loss': 0,
        'dec_loss': 0,
    }

    with torch.no_grad():
        for img, msg in loader:
            img, msg = img.to(args.device), msg.to(args.device)
            batch_stats = net.stats(img, msg)
            for k in stats:
                stats[k] += len(img) * batch_stats[k]

    for k in stats:
        stats[k] = stats[k] / len(dataset)

    print(" ".join([f"{k}: {v:.3f}"for k, v in stats.items()]))

