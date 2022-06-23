import torch
from torchvision import transforms
import os


def device(device_num = 0):
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def net_path(epoch, nz=16, vae=False, numbers=None):
    dir_name = '-'.join(str(n) for n in sorted(numbers)) if numbers else 'all'
    base_dir = mkdir_if_not_exists(f"./trained_net/{'vae' if vae else 'ae'}/{dir_name}")
    return os.path.join(base_dir, f"z{nz:03d}_e{epoch+1:04d}.pth")
