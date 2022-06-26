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

from common import device, mkdir_if_not_exists, mnist_data_root, net_path
from dataset import PartialMNIST
from net import AE, VAE


def positive_rates(input_nums: list[int], val_nums: list[int], threshold: float, epoch: int, vae: bool, nz: int, device: str):
    """ returns positive rates of each class
    Args:
        input_nums(list[int]): which classes the model was trained with.
        val_nums(list[int]): which classes to use in determining threshold.
        threshold(float)
        epoch(int): which epoch the model is trained up to.
        vae(bool):
        nz(int): size of latent 
    """

    valset = PartialMNIST(val_nums, train=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    if vae:
        net = VAE(nz)
    else:
        net = AE(nz)
    net.to(device)
    net.eval()

    net.load_state_dict(torch.load(net_path(epoch - 1, nz, vae, input_nums)))

    criterion = nn.MSELoss(reduction='none')

    all_losses = []
    for images, label in tqdm(valloader, desc='determining threshold'):
        images = images.to(device)
        if vae:
            kl, reconst = net.losses(images)
            loss = kl + reconst
        else:
            output = net(images)
            loss = torch.sum(criterion(output, images), dim=(1, 2, 3))
        all_losses += loss.tolist()

    all_losses.sort()
    threshold = all_losses[int(len(all_losses) * threshold)]

    testset = MNIST(mnist_data_root, train=False, download=True, transform=ToTensor())
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    positive_num = [0] * 10
    num_num = [0] * 10

    for images, labels in tqdm(testloader, desc='calculating positive rate of MNIST classes'):
        images = images.to(device)
        labels = labels.tolist()
        if vae:
            kl, reconst = net.losses(images)
            losses = kl + reconst
        else:
            output = net(images)
            losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

        losses = losses.tolist()

        for loss, label in zip(losses, labels):
            if loss > threshold:
                positive_num[label] += 1
            num_num[label] += 1

    fashionset = FashionMNIST(mnist_data_root, train=False, download=True, transform=ToTensor())
    fashionloader = DataLoader(fashionset, batch_size=64, shuffle=False, num_workers=2)

    positive = 0
    for images, _ in tqdm(fashionloader, desc='calculating positive rate of Fashion-MNIST'):
        images = images.to(device)
        if vae:
            kl, reconst = net.losses(images)
            losses =  kl + reconst
        else:
            output = net(images)
            losses = torch.sum(criterion(output, images), dim=(1, 2, 3))

        positive += torch.count_nonzero(losses > threshold).item()

    positive_rates = np.array(positive_num)/np.array(num_num)
    return np.append(positive_rates, positive/len(fashionset))


if __name__ == "__main__":
    parser = ArgumentParser(description="Anomaly detection when trained with partial MNIST classes.")
    parser.add_argument('inputnums', type=int, nargs="+", help="The model trained by this classes will be used.")
    parser.add_argument('--nepoch', type=int, help="which epoch model to use for anomaly detection", default=50)
    parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
    parser.add_argument('--vae', action="store_true", help="choose vae model")
    parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
    parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
    args = parser.parse_args()

    device = device(args.gpu_num)

    left = np.arange(0, 11)
    label = [str(i) for i in range(10)] + ["Fashion"]
    plt.bar(left, positive_rates(args.inputnums, args.inputnums, args.threshold, args.nepoch, args.vae, args.nz, device), tick_label=label)
    input_name = '-'.join(str(n) for n in sorted(args.inputnums)) if args.inputnums else ''
    path = os.path.join(mkdir_if_not_exists(f'graph/{"v" if args.vae else ""}ae'), f"{'onlykl' if args.kl else ''}{input_name}_t{args.threshold:.3f}.png")
    plt.title(f"Positive rates (Trained classes: {input_name})")
    plt.xlabel('class')
    plt.ylabel('positive rate')
    plt.savefig(path, bbox_inches='tight')
    print(f'Image saved {path}')
