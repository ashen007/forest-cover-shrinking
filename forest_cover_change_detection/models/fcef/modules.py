import torch

from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel=3,
                 batch_norm=True,
                 dropout=True,
                 dropout_rate=0.2,
                 blocks=2):
        super(DownSample, self).__init__()

        layers = []

        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, filters, kernel, padding=1))

            if batch_norm:
                layers.append(nn.BatchNorm2d(filters))

            layers.append(nn.ReLU())

            if dropout:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = filters

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 batch_norm=True,
                 dropout=True,
                 dropout_rate=0.2,
                 blocks=3):
        super(UpSample, self).__init__()

        layers = []

        for _ in range(blocks):
            layers.append(nn.ConvTranspose2d(in_channels, filters, kernel,
                                             stride, padding, output_padding))

            if batch_norm:
                layers.append(nn.BatchNorm2d(filters))

            layers.append(nn.ReLU())

            if dropout:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = filters

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 down_sample=True
                 ):
        super(Residual, self).__init__()

        self.main_branch = nn.Sequential(nn.Conv2d(in_channels, filters, kernel, padding=padding),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2),
                                         nn.Conv2d(filters, filters, kernel, padding=padding),
                                         )
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels, filters, 1, padding=padding),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2)
                                       )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_sc = self.short_cut(x)

        return F.relu(x_main + x_sc)



if __name__ == "__main__":
    t = torch.randn(4, 6, 48, 48)
    t_ = torch.randn(4, 32, 24, 24)
    sub_sample = DownSample(6, 16)
    up_sample = UpSample(32, 16, stride=2, blocks=1)

    print(sub_sample)
    print(up_sample)
    print(sub_sample(t).shape)
    print(up_sample(t_).shape)
