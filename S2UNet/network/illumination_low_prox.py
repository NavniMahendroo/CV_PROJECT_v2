import torch.nn as nn


class Q_Low_ProxNet(nn.Module):
    def __init__(self):
        super(Q_Low_ProxNet, self).__init__()
        self.res_block1 = ResBlock(in_channels=1, out_channels=32)
        self.res_block2 = ResBlock(in_channels=1, out_channels=32)

    def forward(self, x):
        out = self.res_block1(x)
        out = self.res_block2(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2.is_last_conv = True

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.conv1(x))
        out = self.conv2(out)
        out = residual + out
        out = self.relu(out)
        return out
