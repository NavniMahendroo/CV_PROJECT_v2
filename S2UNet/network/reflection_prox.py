import torch
import torch.nn as nn


class P_ProxNet(nn.Module):
    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.entry1 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.body = nn.Sequential(
            ChannelAttentionBlock(width),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            SpatialAttentionBlock(width),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.exit = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(width, 3, 1)
        )
        self.exit[2].is_last_conv = True
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, R):
        residual = R
        x = self.entry1(R)
        x = self.body(x)
        x = self.exit(x)
        out = self.relu2(residual + x)
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        y = (avg_out + max_out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
