import torch
from torch import nn
import torch.nn.functional as F


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
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_enclogvar = nn.Linear(200, z_dim)  # predict log(σ^2)
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)
    
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        logvar = self.dense_enclogvar(x)
        return mean, logvar
    
    def _sample_z(self, mean, logvar):
        #再パラメータ化トリック
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon
 
    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        # 出力が0~1になるようにsigmoid
        x = torch.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, logvar = self._encoder(x)
        z = self._sample_z(mean, logvar)
        x = self._decoder(z)
        return x

    def loss(self, x):
        # Reconstruction + KL divergence losses summed over all elements
        # returns avarage within a batch.
        mean, logvar = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + logvar - mean**2 - logvar.exp()))
        z = self._sample_z(mean, logvar)
        y = self._decoder(z)
        reconstruction = F.binary_cross_entropy(y, x, reduction="sum") / x.size(0)

        return KL , reconstruction

    def losses(self, x):
        # Reconstruction + KL divergence losses summed over elements
        # returns Tensor whose length equals batch size
        mean, logvar = self._encoder(x)
        KL = -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1)
        z = self._sample_z(mean, logvar)
        y = self._decoder(z)
        reconstruction = torch.sum(F.binary_cross_entropy(y, x, reduction="none"), dim=1)
        return KL, reconstruction