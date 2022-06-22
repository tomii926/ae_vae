import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from common import device, net_path, transform
from net import AE

device = device()

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="which epochs to generate image", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('--vae', action="store_true", help="choose vae model")
args = parser.parse_args()

testset = MNIST('.', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

net = AE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(net_path(args.nepoch - 1, args.nz, args.vae)))

images, _ = iter(testloader).__next__()
images = images.to(device)
save_image(images.view(-1, 1, 28, 28), f"images/multi/{'v' if args.vae else ''}ae_z{args.nz:03d}_e{args.nepoch+1:04d}_real.png")
reconst = net(images)
save_image(reconst.view(-1, 1, 28, 28), f"images/multi/{'v' if args.vae else ''}ae_z{args.nz:03d}_e{args.nepoch+1:04d}_fake.png")
