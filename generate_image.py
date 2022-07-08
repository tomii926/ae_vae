import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from common import device, net_path
from net import VAE

device = device()

parser = ArgumentParser(description='Generate images from random latent vectors using the learned model.')
parser.add_argument('--nepoch', type=int, help="number of epochs to generate images", default=200)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
args = parser.parse_args()

net = VAE(args.nz)
net.to(device)
net.eval()

fixed_z = torch.randn(64, args.nz).to(device)

print(f'generating epoch={args.nepoch} image')
net.load_state_dict(torch.load(net_path(args.nepoch-1, args.nz, True)))
y = net._decoder(fixed_z)
save_image(y, f'images/z{args.nz:02d}_e{args.nepoch:04d}_.png', pad_value=1)
