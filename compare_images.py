import os
from argparse import ArgumentParser
from tokenize import Single

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from common import device, net_path
from dataset import transform, SingleMNIST
from net import AE

device = device()

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="which epochs to generate image", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('-i', '--input-nums', type=int, nargs="*", help="if this argument is specified, the model trained by this number(s) will be used.")
parser.add_argument('-t', '--test-nums', type=int, nargs="*", help="which number(s) to use in test.")
args = parser.parse_args()

if args.test_nums is None:
    testset = MNIST('.', train=False, download=True, transform=transform)
else:
    testset = SingleMNIST(args.test_nums, False)

testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


net = AE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(net_path(args.nepoch - 1, args.nz, args.vae, args.input_nums)))

images, _ = iter(testloader).__next__()
images = images.to(device)

def image_path(epoch, nz, vae, real, input_numbers, test_numbers):
    input_name = '-'.join(str(n) for n in sorted(input_numbers)) if input_numbers else ''
    test_name = '-'.join(str(n) for n in sorted(test_numbers)) if test_numbers else ''
    return f"images/{'v' if vae else ''}ae_{input_name}_{test_name}_z{nz:03d}_e{epoch+1:04d}_{'real' if real else 'fake'}.png"

save_image(images.view(-1, 1, 28, 28), image_path(args.nepoch - 1, args.nz, args.vae, True, args.input_nums, args.test_nums))
reconst = net(images)
save_image(reconst.view(-1, 1, 28, 28), image_path(args.nepoch - 1, args.nz, args.vae, False, args.input_nums, args.test_nums))

