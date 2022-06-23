import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from common import device, mkdir_if_not_exists, net_path
from dataset import SingleMNIST
from net import AE, VAE

device = device()

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="which epochs to generate image", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('-i', '--input-nums', type=int, nargs="*", help="if this argument is specified, the model trained by this number(s) will be used.")
parser.add_argument('-v', '--valid-nums', type=int, nargs="*", help="which classes to use in determining threshold. if not specified, this will be the same as input-nums")
parser.add_argument('--kl', action="store_true", help="use only KL divergence when determining threshold.")
parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
args = parser.parse_args()


if args.valid_nums is None:
    valid_nums = args.input_nums
else:
    valid_nums = args.valid_nums

valset = SingleMNIST(valid_nums, True)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

if args.vae:
    net = VAE(args.nz)
else:
    net = AE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(net_path(args.nepoch - 1, args.nz, args.vae, args.input_nums)))

criterion = nn.MSELoss(reduction='none')

print('determining threshold...')

all_losses = []
for images, label in tqdm(valloader):
    images = images.to(device)
    if args.vae:
        kl, reconst = net.losses(images)
        loss = reconst if args.kl else kl + reconst
    else:
        output = net(images)
        loss = torch.sum(criterion(output, images), dim=(1, 2, 3))
    all_losses += loss.tolist()

all_losses.sort()
threshold = all_losses[int(len(all_losses) * args.threshold)]

print(f"threshold = {threshold}")

testset = MNIST('.', train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

true_false_positive_case_num = [0] * 10
num_num = [0] * 10

print('calculating false/true positive rate...')
for images, labels in tqdm(testloader):
    images = images.to(device)
    labels = labels.tolist()
    if args.vae:
        kl, reconst = net.losses(images)
        losses = reconst if args.kl else kl + reconst
    else:
        output = net(images)
        losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

    losses = losses.tolist()

    for loss, label in zip(losses, labels):
        if loss > threshold:
            true_false_positive_case_num[label] += 1
        num_num[label] += 1

fashionset = FashionMNIST('.', train=False, download=True, transform=ToTensor())
fashionloader = DataLoader(fashionset, batch_size=64, shuffle=False, num_workers=2)

positive = 0
for images, _ in tqdm(fashionloader):
    images = images.to(device)
    if args.vae:
        kl, reconst = net.losses(images)
        losses = reconst if args.kl else kl + reconst
    else:
        output = net(images)
        losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

    losses = losses.tolist()

    for loss in losses:
        if loss > threshold:
            positive += 1

true_false_positive_rates = np.array(true_false_positive_case_num)/np.array(num_num)
true_false_positive_rates = np.append(true_false_positive_rates, positive/len(fashionset))
print(true_false_positive_rates)

left = np.arange(0, 11)
label = [str(i) for i in range(10)] + ["Fashion"]
plt.bar(left, true_false_positive_rates, tick_label=label)
input_name = '-'.join(str(n) for n in sorted(args.input_nums)) if args.input_nums else ''
val_name = '-'.join(str(n) for n in sorted(args.valid_nums)) if args.valid_nums else ''
path = os.path.join(mkdir_if_not_exists(f'tables/{"v" if args.vae else ""}ae'), f"{'onlykl' if args.kl else ''}{input_name}_{val_name}_t{args.threshold:.3f}.png")
plt.savefig(path)






    





