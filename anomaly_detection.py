import os
from argparse import ArgumentParser
from tokenize import Single

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from common import device, mkdir_if_not_exists, net_path
from dataset import SingleMNIST, transform
from net import AE, VAE
import numpy as np
from tqdm import tqdm

device = device()

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="which epochs to generate image", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('-i', '--input-nums', type=int, nargs="*", help="if this argument is specified, the model trained by this number(s) will be used.")
parser.add_argument('-v', '--valid-nums', type=int, nargs="*", help="which classes to use in determining threshold. if not specified, this will be the same as input-nums")
parser.add_argument('--image-num', type=int, help="how many images to compare.", default=64)
args = parser.parse_args()


if args.valid_nums is None:
    valid_nums = args.input_nums
else:
    valid_nums = args.valid_nums

valset = SingleMNIST(valid_nums, False)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

if args.vae:
    net = VAE(args.nz)
else:
    net = AE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(net_path(args.nepoch - 1, args.nz, args.vae, args.input_nums)))

criterion = nn.MSELoss()

print('determining threshold...')

losses = []
for images, label in tqdm(valloader):
    images = images.to(device)
    if args.vae:
        kl, reconst = net.loss(images)
        loss = kl + reconst
    else:
        output = net(images)
        loss = criterion(output, images)
    losses.append(loss.item())
    # print(loss.item())

losses.sort()
threshold = losses[int(len(losses) * 0.9)]

print(f"threshold = {threshold}")

testset = MNIST('.', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

true_false_positive_case_num = [0] * 10
num_num = [0] * 10


print('calculating false/true positive rate...')
for image, label in tqdm(testloader):
    image = image.to(device)
    label = label.item()
    if args.vae:
        kl, reconst = net.loss(image)
        loss = kl + reconst
    else:
        output = net(image)
        loss = criterion(output, image).item()

    # print(loss, label)
    if loss > threshold:
        true_false_positive_case_num[label] += 1
    num_num[label] += 1

true_false_positive_rates = np.array(true_false_positive_case_num)/np.array(num_num)
print(true_false_positive_rates)






    





