import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from common import device, mkdir_if_not_exists, mnist_data_root, net_path
from dataset import PartialMNIST
from net import AE, VAE

parser = ArgumentParser(description="Train model")
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('--vae', action="store_true", help="train VAE model")
parser.add_argument('-i', '--input-nums', type=int, nargs="*", help="Classes used for training model")
args = parser.parse_args()

device = device(args.gpu_num)

max_epoch=args.nepoch
batch_size=64

transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

if args.input_nums is not None:
    trainset = PartialMNIST(args.input_nums, True, transform=transform)
else:
    trainset = MNIST(root=mnist_data_root, train=True, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valset = MNIST(mnist_data_root, train=False, download=True, transform=transforms.ToTensor())
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

train_loss_series = []
val_loss_series = []

if args.vae:
    vae = VAE(args.nz)
    vae.to(device)

    optimizer = optim.Adam(vae.parameters())
    
    for epoch in range(max_epoch):
        losses = []
        kl_losses = []
        rec_losses = []

        for images, _ in trainloader:
            images = images.to(device)

            optimizer.zero_grad()

            KL_loss, reconstruction_loss = vae.loss(images)
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            kl_losses.append(KL_loss.item())
            rec_losses.append(reconstruction_loss.item())

        vae.eval()
        val_losses = []
        for images, _ in valloader:
            images = images.to(device)
            KL_loss, reconstruction_loss = vae.loss(images)
            loss = KL_loss + reconstruction_loss
            val_losses.append(loss.item())

        print(f'epoch: {epoch + 1} reconstruction:{np.mean(rec_losses)} KL: {np.mean(kl_losses)}  Train Lower Bound: {np.mean(losses)}, Val Lower Bound: {np.mean(val_losses)}')
        torch.save(vae.state_dict(), net_path(epoch, args.nz, True, numbers=args.input_nums))

        train_loss_series.append(np.mean(losses))
        val_loss_series.append(np.mean(val_losses))


else: # autoencoder
    ae = AE(args.nz)
    ae.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters())

    for epoch in range(max_epoch,):
        ae.train()
        losses = []
        for images, _ in trainloader:
            images = images.to(device)

            optimizer.zero_grad()

            x_output = ae(images)

            loss = criterion(x_output, images)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        ae.eval()
        val_losses = []
        for images, _ in valloader:
            images = images.to(device)
            output = ae(images)
            loss = criterion(output, images)
            val_losses.append(loss.item())

        print(f'epoch: {epoch + 1}  Train loss: {np.mean(losses)}  Val loss: {np.mean(val_losses)}')
        torch.save(ae.state_dict(), net_path(epoch, args.nz, False, numbers=args.input_nums))

        train_loss_series.append(np.mean(losses))
        val_loss_series.append(np.mean(val_losses))


print('Finished Training')

plt.title('learning curve')
x = np.arange(0, args.nepoch)
plt.plot(x, train_loss_series, label="train")
plt.plot(x, val_loss_series, label="val")
plt.ylabel('Loss')
plt.xlabel('Epoch')

image_path = mkdir_if_not_exists(f'./graph/{"v" if args.vae else ""}ae')
plt.savefig(os.path.join(image_path, f'learning_curve_nz{args.nz:02d}.png'), bbox_inches='tight')

