from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm, trange

from common import device, net_path
from dataset import SingleMNIST, transform
from net import AE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-i', '--input-num', type=int, help="if this argument is specified, the model will be trained by only this number.", required=False)
args = parser.parse_args()

device = device(args.gpu_num)

max_epoch=args.nepoch
batch_size=64

if args.input_num is not None:
    trainset = SingleMNIST(args.input_num, True)
else:
    trainset = MNIST(root='.', train=True, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

ae = AE(args.nz)
ae.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters())

for epoch in trange(max_epoch, desc="epoch"):
    for images, _ in tqdm(trainloader, leave=False, desc="batch"):
        images = images.to(device)

        optimizer.zero_grad()

        x_output = ae(images)

        loss = criterion(x_output, images)
        loss.backward()
        optimizer.step()

    torch.save(ae.state_dict(), net_path(epoch, number=args.input_num))
    
print('Finished Training')
