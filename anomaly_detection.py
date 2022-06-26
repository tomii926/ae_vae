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

parser = ArgumentParser(description="Anomaly detection when trained with partial MNIST classes.")
parser.add_argument('inputnums', type=int, nargs="+", help="The model trained by this classes will be used.")
parser.add_argument('--nepoch', type=int, help="Which epochs to use for anomaly detection", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('--no-kl', action="store_true", help="KL divergence is not used in determining the threshold.")
parser.add_argument('--kl', action="store_true", help="Only KL divergence is used when determining threshold.")
parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = device(args.gpu_num)

valset = SingleMNIST(args.inputnums, train=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

if args.vae:
    net = VAE(args.nz)
else:
    net = AE(args.nz)
net.to(device)
net.eval()

net.load_state_dict(torch.load(net_path(args.nepoch - 1, args.nz, args.vae, args.inputnums)))

criterion = nn.MSELoss(reduction='none')

print('determining threshold...')

all_losses = []
for images, label in tqdm(valloader):
    images = images.to(device)
    if args.vae:
        kl, reconst = net.losses(images)
        loss = kl if args.kl else reconst if args.no_kl else kl + reconst
    else:
        output = net(images)
        loss = torch.sum(criterion(output, images), dim=(1, 2, 3))
    all_losses += loss.tolist()

all_losses.sort()
threshold = all_losses[int(len(all_losses) * args.threshold)]

print(f"threshold = {threshold}")

testset = MNIST('.', train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

positive_num = [0] * 10
num_num = [0] * 10

print('calculating positive rate of MNIST classes')
for images, labels in tqdm(testloader):
    images = images.to(device)
    labels = labels.tolist()
    if args.vae:
        kl, reconst = net.losses(images)
        losses = kl if args.kl else reconst if args.no_kl else kl + reconst
    else:
        output = net(images)
        losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

    losses = losses.tolist()

    for loss, label in zip(losses, labels):
        if loss > threshold:
            positive_num[label] += 1
        num_num[label] += 1

print('calculating positive rate of Fashion-MNIST')
fashionset = FashionMNIST('.', train=False, download=True, transform=ToTensor())
fashionloader = DataLoader(fashionset, batch_size=64, shuffle=False, num_workers=2)

positive = 0
for images, _ in tqdm(fashionloader):
    images = images.to(device)
    if args.vae:
        kl, reconst = net.losses(images)
        losses = kl if args.kl else reconst if args.no_kl else kl + reconst
    else:
        output = net(images)
        losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

    positive += torch.count_nonzero(losses > threshold).item()

positive_rates = np.array(positive_num)/np.array(num_num)
positive_rates = np.append(positive_rates, positive/len(fashionset))

print(positive_rates)

left = np.arange(0, 11)
label = [str(i) for i in range(10)] + ["Fashion"]
plt.bar(left, positive_rates, tick_label=label)
input_name = '-'.join(str(n) for n in sorted(args.inputnums)) if args.inputnums else ''
path = os.path.join(mkdir_if_not_exists(f'tables/{"v" if args.vae else ""}ae'), f"{'onlykl' if args.kl else ''}{input_name}_t{args.threshold:.3f}.png")
plt.savefig(path)
