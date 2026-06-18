import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainCNNVAE(nn.Module):

    def __init__(self, in_ch=1, latent_dim=128):
        super().__init__()

        # ==========================================
        # Encoder
        # ==========================================
        self.encoder = nn.Sequential(

            # (401,64)
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (200,32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (100,16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (50,8)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (25,4)
        )

        # ==========================================
        # latent
        # ==========================================
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # ==========================================
        # decoder input
        # ==========================================
        self.fc_dec = nn.Linear(latent_dim, 512 * 25 * 4)

        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512, 256,
                kernel_size=2,
                stride=2
            ),  # (50,8)

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=2,
                stride=2
            ),  # (100,16)

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),  # (200,32)

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 1,
                kernel_size=2,
                stride=2
            )   # (400,64)
        )

    # ==========================================
    # encode
    # ==========================================
    def encode(self, x):
        h = self.encoder(x)
        # (B,512,25,4)
        pooled = self.global_pool(h)
        # (B,512,1,1)
        pooled = pooled.view(x.size(0), -1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar

    # ==========================================
    # reparameterization
    # ==========================================
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    # ==========================================
    # decode
    # ==========================================
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 512, 25, 4)
        y = self.decoder(h)
        # strict alignment
        y = F.interpolate(
            y,
            size=(401, 64),
            mode="bilinear",
            align_corners=False
        )

        return y

    # ==========================================
    # forward
    # ==========================================
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)

        return y, mu, logvar

    # ==========================================
    # loss
    # ==========================================
    def loss_function(
        self,
        x,
        y,
        mu,
        logvar,
        beta=1.0
    ):

        # reconstruction
        rec_loss = F.mse_loss(y, x)
        # KL
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        total_loss = rec_loss + beta * kl_loss

        return total_loss, rec_loss, kl_loss