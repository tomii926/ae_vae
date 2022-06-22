import torch
from torchvision import transforms


def device(device_num = 0):
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])


def net_path(epoch, nz=20, vae=False, multi_class=True):
    return f"./trained_net/{'multi' if multi_class else 'False'}/{'v' if vae else ''}ae_z{nz:03d}_e{epoch+1:04d}.pth"
