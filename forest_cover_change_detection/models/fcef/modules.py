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

            layers.append(nn.LeakyReLU())

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

            layers.append(nn.LeakyReLU())

            if dropout:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = filters

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualDownSample(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel=3,
                 padding=1,
                 down_sample=False
                 ):
        super(ResidualDownSample, self).__init__()

        if not down_sample:
            self.main_branch = nn.Sequential(nn.Conv2d(in_channels, filters, kernel, padding=padding),
                                             nn.BatchNorm2d(filters),
                                             nn.LeakyReLU(),
                                             nn.Conv2d(filters, filters, kernel, padding=padding),
                                             nn.BatchNorm2d(filters)
                                             )
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, filters, 1),
                                           nn.BatchNorm2d(filters)
                                           )

        else:
            self.main_branch = nn.Sequential(nn.Conv2d(in_channels, filters, kernel, padding=padding),
                                             nn.BatchNorm2d(filters),
                                             nn.LeakyReLU(),
                                             nn.MaxPool2d(2),
                                             nn.Conv2d(filters, filters, kernel, padding=padding),
                                             nn.BatchNorm2d(filters)
                                             )
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, filters, 1),
                                           nn.BatchNorm2d(filters),
                                           nn.MaxPool2d(2)
                                           )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_sc = self.short_cut(x)

        return F.leaky_relu(x_main + x_sc)


class ResidualUpSample(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel=3,
                 padding=1,
                 out_padding=1,
                 stride=2
                 ):
        super(ResidualUpSample, self).__init__()

        self.main_branch = nn.Sequential(nn.ConvTranspose2d(in_channels, filters, kernel, stride, padding, out_padding),
                                         nn.BatchNorm2d(filters),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(filters, filters, kernel, padding=padding),
                                         nn.BatchNorm2d(filters)
                                         )
        self.short_cut = nn.Sequential(nn.ConvTranspose2d(in_channels, filters, kernel, stride, padding, out_padding),
                                       nn.BatchNorm2d(filters)
                                       )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_sc = self.short_cut(x)

        return F.leaky_relu(x_main + x_sc)


class ResNeXtDownSample(nn.Module):

    def __init__(self, in_channels, out_channels, down_sample=False):
        super(ResNeXtDownSample, self).__init__()

        sub_net_layers = [nn.Conv2d(in_channels, 4, 1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.Conv2d(4, 4, 3, padding=1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.Conv2d(4, in_channels, 1)]

        if down_sample:
            sub_net_layers.append(nn.MaxPool2d(2))

        self.path = nn.Sequential(*sub_net_layers)
        self.out = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU())
        self.pool = nn.MaxPool2d(2)
        self.down_smp = down_sample

    def forward(self, x):
        xs = [self.path(x) for _ in range(4)]
        early_agg = xs[0]

        for x_ in xs[1:]:
            early_agg += x_

        if self.down_smp:
            x = self.pool(x)

        late_agg = F.leaky_relu(F.leaky_relu(early_agg) + x)

        return self.out(late_agg)


class ResNeXtUpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNeXtUpSample, self).__init__()

        sub_net_layers = [nn.Conv2d(in_channels, 4, 1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.ConvTranspose2d(4, 4, 3, 2, 1, 1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.Conv2d(4, in_channels, 1)]

        self.path = nn.Sequential(*sub_net_layers)
        self.out = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU())
        self.short_cut = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1)

    def forward(self, x):
        xs = [self.path(x) for _ in range(4)]
        early_agg = xs[0]

        for x_ in xs[1:]:
            early_agg += x_

        late_agg = F.leaky_relu(F.leaky_relu(early_agg) + self.short_cut(x))

        return self.out(late_agg)


class SEBlock(nn.Module):

    def __init__(self, in_channels, reducer=4):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(nn.Linear(in_channels, in_channels // reducer),
                                      nn.LeakyReLU(),
                                      nn.Linear(in_channels // reducer, in_channels),
                                      nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        x_ = self.pool(x).view(b, c)
        x_ = self.se_block(x_).view(b, c, 1, 1)

        return x * x_.expand_as(x)


if __name__ == "__main__":
    t = torch.randn(4, 16, 48, 48)
    t_ = torch.randn(4, 16, 24, 24)
    # sub_sample = DownSample(6, 16)
    # up_sample = UpSample(32, 16, stride=2, blocks=1)
    # residual = ResidualDownSample(6, 16, down_sample=True)
    # residual_ = ResidualUpSample(16, 32)
    # resnext = ResNeXtDownSample(16, down_sample=True)
    # resnext_ = ResNeXtUpSample(16, 16)
    se = SEBlock(16)

    # print(sub_sample)
    # print(up_sample)
    # print(residual)
    # print(residual_)
    # print(sub_sample(t).shape)
    # print(up_sample(t_).shape)
    # print(residual(t).shape)
    # print(residual_(t_).shape)
    # print(resnext_(t_).shape)
    print(se(t).shape)
