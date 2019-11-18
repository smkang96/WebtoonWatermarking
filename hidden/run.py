import argparse


def make_parser(modes):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    for name, func in modes.items():
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(func=func)

    return parser, subparsers.choices

from train import train
from test import test
from save_img import save_img

modes = {
    'train': train,
    'test': test,
    'save_img': save_img,
}
parser, subparsers = make_parser(modes)

for mode, subparser in subparsers.items():
    subparser.add_argument('--img_size', type=int)
    subparser.add_argument('--msg_l', type=int)
    subparser.add_argument('--device', type=int)
    subparser.add_argument('--test_device', type=int)
    subparser.add_argument('--batch_size', type=int)
    subparser.add_argument('--noise_type', type=str)

    if 'save_img' not in mode:
        subparser.add_argument('--enc_scale', type=float)
        subparser.add_argument('--dec_scale', type=float)
        subparser.add_argument('--adv_scale', type=float)

    if 'train' in mode:
        subparser.add_argument('--epochs', type=int)
        subparser.add_argument('--save_freq', type=int)
        subparser.add_argument('--test_freq', type=int)

    if 'save_img' == mode:
        subparser.add_argument('--n_imgs', type=int)

args = parser.parse_args()

if args.test_device is None:
    args.test_device = args.device

args.func(args)

