import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.watermark import Watermark
from net import DFW, max_depth
import common.path as path


log_filename = './test.log'


class DFWTest(DFW):
    def __init__(self, args, data):
        super().__init__(args, data)

        self.enc_scale, self.dec_scale = args.enc_scale, args.dec_scale

    def stats(self, img, msg):
        self.eval()

        watermark = self.encoder(msg)
        encoded_img = (img + watermark).clamp(-1, 1)
        noised_img = self.noiser(encoded_img)
        decoded_msg = self.decoder(noised_img)

        enc_loss = torch.norm(watermark, p=2, dim=(1, 2, 3)).mean()
        dec_loss = F.binary_cross_entropy_with_logits(decoded_msg, msg)
        loss = self.enc_scale*enc_loss + self.dec_scale*dec_loss
        
        pred = (torch.sigmoid(decoded_msg) > 0.5).int()
        correct = (pred == msg).sum(1)
        accuracy0 = (correct == self.l).float().mean()
        accuracy3 = (correct > (self.l - 3)).float().mean()

        return {
            'loss': loss.item(),
            'enc_loss': enc_loss.item(),
            'dec_loss': dec_loss.item(),
            'accuracy0': accuracy0,
            'accuracy3': accuracy3,
            'avg_acc': correct.float().mean() / self.l
        }


def test_worker(args, queue):
    log_file = open(log_filename, 'w+', buffering=1)

    dataset = Watermark(args.img_size, args.msg_l, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    net = DFWTest(args, dataset).to(args.test_device)
    net.set_depth(max_depth)
    
    while True:
        epoch_i, state_dict = queue.get()
        net.load_state_dict(state_dict)

        stats = {
            'loss': 0,
            'enc_loss': 0,
            'dec_loss': 0,
            'accuracy0': 0,
            'accuracy3': 0,
            'avg_acc': 0,
        }

        with torch.no_grad():
            for img, msg in loader:
                img, msg = img.to(args.test_device), msg.to(args.test_device)
                batch_stats = net.stats(img, msg)
                for k in stats:
                    stats[k] += len(img) * batch_stats[k]

        for k in stats:
            stats[k] = stats[k] / len(dataset)

        log_file.write("Epoch {} | {}\n".format(epoch_i,  " ".join([f"{k}: {v:.3f}"for k, v in stats.items()])))
        queue.task_done()


def test(args):
    dataset = Watermark(args.img_size, args.msg_l, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    net = DFWTest(args, dataset).to(args.device)
    net.set_depth(max_depth)
    
    net.load_state_dict(torch.load(path.save_path))
    stats = {
        'loss': 0,
        'enc_loss': 0,
        'dec_loss': 0,
        'accuracy0': 0,
        'accuracy3': 0,
        'avg_acc': 0,
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
