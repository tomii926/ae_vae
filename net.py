import torch
from torch import nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 28, 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),            
            nn.MaxPool2d(2)  # b, 64, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 14, 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),           
            nn.MaxPool2d(2)  # b, 128, 7, 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256, 7, 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # b, 256, 3, 3
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=0),  # b, 512, 1, 1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.z = nn.Linear(512, z_dim)  # b, 512 ==> b, latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),# b, latent_dim ==> b, 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding = 0),  # b, 256, 3, 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=3, padding = 1),  # b, 128, 7, 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding = 1),  # b, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding = 1),  # b, 3, 28, 28
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def _encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,512)
        z = self.z(x)
        return z

    def _decoder(self, z):
        x = self.decoder(z)
        x = x.view(-1,512,1,1)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        return x

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

# VAEモデルの実装
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 28, 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),            
            nn.MaxPool2d(2)  # b, 64, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 14, 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),           
            nn.MaxPool2d(2)  # b, 128, 7, 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256, 7, 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # b, 256, 3, 3
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=0),  # b, 512, 1, 1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.mean = nn.Linear(512, z_dim)  # b, 512 ==> b, latent_dim
        
        self.logvar = nn.Linear(512, z_dim)  # b, 512 ==> b, latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),# b, latent_dim ==> b, 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding = 0),  # b, 256, 3, 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=3, padding = 1),  # b, 128, 7, 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding = 1),  # b, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding = 1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def _encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,512)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar
    
    def _sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon
 
    def _decoder(self, z):
        x = self.decoder(z)
        x = x.view(-1,512,1,1)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
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
        KL = -0.5 * torch.mean(torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1))
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
        reconstruction = torch.sum(F.binary_cross_entropy(y, x, reduction="none"), dim=(1, 2, 3))
        return KL, reconstruction


if __name__ == "__main__":
    from torchinfo import summary
    summary(AE(16), (64, 1, 28, 28))

    summary(VAE(16), (64, 1, 28, 28))

