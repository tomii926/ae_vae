from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm, trange

from common import device, net_path
from dataset import SingleMNIST, transform
from net import AE, VAE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('-i', '--input-nums', type=int, nargs="*", help="if this argument is specified, the model will be trained by only this number.")
args = parser.parse_args()

device = device(args.gpu_num)

max_epoch=args.nepoch
batch_size=64

if args.input_nums is not None:
    trainset = SingleMNIST(args.input_nums, True)
else:
    trainset = MNIST(root='.', train=True, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

if args.vae:
    vae = VAE(args.nz)
    vae.to(device)

    optimizer = optim.Adam(vae.parameters())
    
    for epoch in range(max_epoch):
        losses = []
        for images, _ in trainloader:
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = vae.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'epoch: {epoch + 1}  Train Lower Bound: {sum(losses)/len(losses)}')
        torch.save(vae.state_dict(), net_path(epoch, args.nz, True, numbers=args.input_nums))


else: # autoencoder
    ae = AE(args.nz)
    ae.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters())

    for epoch in range(max_epoch,):
        losses = []
        for images, _ in trainloader:
            images = images.to(device)

            optimizer.zero_grad()

            x_output = ae(images)

            loss = criterion(x_output, images)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())


        print(f'epoch: {epoch + 1}  Train Lower Bound: {sum(losses)/len(losses)}')
        torch.save(ae.state_dict(), net_path(epoch, args.nz, False, numbers=args.input_nums))
    
print('Finished Training')
