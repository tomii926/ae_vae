from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from common import device, transform, net_path
from net import AE

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="number of epochs to train for", default=25)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = device(args.gpu_num)

max_epoch=args.nepoch
batch_size=64

trainset = MNIST(root='.', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

ae = AE(args.nz)
ae.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters())

for epoch in range(max_epoch):
    running_loss = 0.0
    for i, (images, _) in enumerate(trainloader, 0):
        images = images.to(device)

        optimizer.zero_grad()

        x_output = ae(images)

        loss = criterion(x_output, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.5f}')
            running_loss = 0.0

    print(f'Model saving ... at {epoch+1}')
    torch.save(ae.state_dict(), net_path(epoch))
    
print('Finished Training')
