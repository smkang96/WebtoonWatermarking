import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.watermark import Watermark
from net import DFW, max_depth
import common.path as path
from utils import HammingCoder
import numpy as np
import pickle
from tqdm import tqdm

log_filename = './test.log'


class DFWTest(DFW):
    def __init__(self, args, data):
        super().__init__(args, data)

        self.enc_scale, self.dec_scale = args.enc_scale, args.dec_scale
        self.hamming_coder = HammingCoder(device=args.device)
              
    def stats(self, img, msg):
        self.eval()
        hamming_msg = torch.stack([self.hamming_coder.encode(x) for x in msg])
        watermark = self.encoder(hamming_msg)
        encoded_img = (img + watermark).clamp(-1, 1)
        noised_img, _ = self.noiser([encoded_img, img])
        decoded_msg_logit = self.decoder(noised_img)
        
        
        enc_loss = torch.norm(watermark, p=2, dim=(1, 2, 3)).mean()
        dec_loss = F.binary_cross_entropy_with_logits(decoded_msg_logit, hamming_msg)
        loss = self.enc_scale*enc_loss + self.dec_scale*dec_loss
        
        pred_without_hamming_dec = (torch.sigmoid(decoded_msg_logit) > 0.5).int()
        pred_msg = torch.stack([self.hamming_coder.decode(x) for x in pred_without_hamming_dec])
        correct = (pred_msg == msg).sum(1)
        accuracy0 = (correct == self.l).float().mean()
        accuracy3 = (correct > (self.l - 3)).float().mean()
        num_right_bits_without_hamming = ((pred_without_hamming_dec == hamming_msg).sum(1)).float().mean()
        return {
            'loss': loss.item(),
            'enc_loss': enc_loss.item(),
            'dec_loss': dec_loss.item(),
            'accuracy0': accuracy0.item(),
            'accuracy3': accuracy3.item(),
            'num_right_bits_without_hamming': num_right_bits_without_hamming.item(),
            'avg_acc': (correct.float().mean() / self.l).item(),
            'num_right_bits': correct.float().mean().item()
        }
        

def test_worker(args, queue):
    log_file = open(log_filename, 'w+', buffering=1)

    dataset = Watermark(args.img_size, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    msg_dist = torch.distributions.Bernoulli(probs=0.5*torch.ones(args.msg_l))
    
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
            'num_right_bits_without_hamming': 0,
            'avg_acc': 0,
            'num_right_bits': 0
        }

        with torch.no_grad():
            for img in loader:
                msg = msg_dist.sample([img.shape[0]])
                img, msg = img.to(args.test_device), msg.to(args.test_device)
                batch_stats = net.stats(img, msg)
                for k in stats:
                    stats[k] += len(img) * batch_stats[k]

        for k in stats:
            stats[k] = stats[k] / len(dataset)

        log_file.write("Epoch {} | {}\n".format(epoch_i,  " ".join([f"{k}: {v:.3f}"for k, v in stats.items()])))
        queue.task_done()

def test_per_user(args):
    dataset = Watermark(args.img_size, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    msg_dist = torch.distributions.Bernoulli(probs=0.5*torch.ones(args.msg_l))
    list_msg = msg_dist.sample([args.n_users])
    
    np.savetxt("./foo.csv", list_msg.numpy(), delimiter=",")
    net = DFWTest(args, dataset).to(args.device)
    net.set_depth(max_depth)
    net.load_state_dict(torch.load(path.save_path))
    list_stats = []
    with torch.no_grad():
        for i, msg in tqdm(enumerate(list_msg)):
            stats = {
                'loss': 0,
                'enc_loss': 0,
                'dec_loss': 0,
                'accuracy0': 0,
                'accuracy3': 0,
                'num_right_bits_without_hamming': 0,
                'avg_acc': 0,
                'num_right_bits': 0
            }
            for img in loader:
                msg_batched = msg.repeat(img.shape[0], 1)
                img, msg_batched = img.to(args.device), msg_batched.to(args.device)
                batch_stats = net.stats(img, msg_batched)
                for k in stats:
                    stats[k] += len(img) * batch_stats[k]

            for k in stats:
                stats[k] = stats[k] / len(dataset)
            list_stats.append(stats)
            #print("User", i, "Noise type:", args.noise_type, " ".join([f"{k}: {v:.3f}"for k, v in stats.items()]))
    pickle.dump( list_stats, open( "list_stats.p", "wb" ) )
    
    
def test(args):
    dataset = Watermark(args.img_size, train=False, dev=False)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    msg_dist = torch.distributions.Bernoulli(probs=0.5*torch.ones(args.msg_l))
    
    net = DFWTest(args, dataset).to(args.device)
    net.set_depth(max_depth)
    
    net.load_state_dict(torch.load(path.save_path))
    stats = {
        'loss': 0,
        'enc_loss': 0,
        'dec_loss': 0,
        'accuracy0': 0,
        'accuracy3': 0,
        'num_right_bits_without_hamming': 0,
        'avg_acc': 0,
        'num_right_bits': 0
    }

    with torch.no_grad():
        for img in loader:
            msg = msg_dist.sample([img.shape[0]])
            img, msg = img.to(args.device), msg.to(args.device)
            batch_stats = net.stats(img, msg)
            for k in stats:
                stats[k] += len(img) * batch_stats[k]

    for k in stats:
        stats[k] = stats[k] / len(dataset)
    print("Noise type: ", args.noise_type, " ".join([f"{k}: {v:.3f}"for k, v in stats.items()]))
