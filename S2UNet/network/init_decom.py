import torch.nn as nn


class InitialDecomposer(nn.Module):
    def __init__(self, in_ch=3, width=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.feature_enhancer = nn.Sequential(
            LightweightS2Attention(width),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.r_branch = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(width, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.l_branch = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(width, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        base_feat = self.encoder(x)
        enhanced_feat = self.feature_enhancer(base_feat)
        R = self.r_branch(enhanced_feat)
        L = self.l_branch(enhanced_feat)

        return R, L


class LightweightS2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spectral_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.spectral_att(x.mean(dim=(2, 3), keepdim=True)) * self.spatial_att(x)
