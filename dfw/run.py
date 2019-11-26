import argparse


def make_parser(modes):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    for name, func in modes.items():
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(func=func)

    return parser, subparsers.choices

from train_with_eval import pretrain, train
from test import test, test_per_user
from save_img import save_img

modes = {
    'pretrain': pretrain,
    'train': train,
    'test': test,
    'test_per_user': test_per_user,
    'save_img': save_img,
}
parser, subparsers = make_parser(modes)

for mode, subparser in subparsers.items():
    subparser.add_argument('--img_dir', type=str)
    subparser.add_argument('--img_size', type=int)
    subparser.add_argument('--msg_l', type=int)
    subparser.add_argument('--device', type=int)
    subparser.add_argument('--test_device', type=int)
    subparser.add_argument('--noise_type', type=str)

    if 'train' == mode:
        subparser.add_argument('--epochs', type=int)
        subparser.add_argument('--save_freq', type=int)
        subparser.add_argument('--test_freq', type=int)
        subparser.add_argument('--annealing_epochs', type=int)

    if 'save_img' == mode:
        subparser.add_argument('--n_imgs', type=int)

    if 'save_img' not in mode:
        subparser.add_argument('--batch_size', type=int)
        subparser.add_argument('--enc_scale', type=float)
        subparser.add_argument('--dec_scale', type=float)

args = parser.parse_args()

if args.test_device is None:
    args.test_device = args.device

args.func(args)

