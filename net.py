from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F


# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


class AE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_enc3 = nn.Linear(200, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        x = self.dense_enc3(x)
        return x

    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = torch.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

# VAEモデルの実装
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        # Encoder, xを入力にガウス分布のパラメータmu, sigmaを出力
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_encvar = nn.Linear(200, z_dim)
        # Decoder, zを入力にベルヌーイ分布のパラメータlambdaを出力
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)
    
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        std = F.softplus(self.dense_encvar(x))
        return mean, std
    
    def _sample_z(self, mean, std):
        #再パラメータ化トリック
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon
 
    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        # 出力が0~1になるようにsigmoid
        x = torch.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, std = self._encoder(x)
        z = self._sample_z(mean, std)
        x = self._decoder(z)
        return x, z

    def loss(self, x):
        mean, std = self._encoder(x)

        # sum latent vector's KL divergence, then calculate avarage in a batch.
        KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1))
    
        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # sum every pixel's bce loss, then calculate avarage in a batch.
        reconstruction = F.binary_cross_entropy(y, x, reduction="sum") / x.size()[0]

        return KL, reconstruction 