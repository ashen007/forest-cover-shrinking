import torch

from torch import nn
from torch.nn import functional as F
from forest_cover_change_detection.layers.activation_ import RSoftMax


class BaseFeatureExtractor(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseFeatureExtractor, self).__init__()

    def forward(self, x):
        return


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
                 out_channels,
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
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel,
                                             stride, padding, output_padding))

            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.LeakyReLU())

            if dropout:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = out_channels

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualDownSample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3,
                 padding=1):
        super(ResidualDownSample, self).__init__()

        self.main_branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
                                         nn.BatchNorm2d(out_channels),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(out_channels, out_channels, 1),
                                         nn.BatchNorm2d(out_channels)
                                         )
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels)
                                       )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_sc = self.short_cut(x)

        return F.leaky_relu(x_main + x_sc)


class ResidualUpSample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3):
        super(ResidualUpSample, self).__init__()

        self.main_branch = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(out_channels, out_channels, kernel, padding=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.LeakyReLU(),
                                         nn.ConvTranspose2d(out_channels, out_channels, 1),
                                         nn.BatchNorm2d(out_channels)
                                         )
        self.short_cut = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels)
                                       )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_sc = self.short_cut(x)

        return F.leaky_relu(x_main + x_sc)


class ResNeXtDownSample(nn.Module):

    def __init__(self, in_channels, out_channels, c=32, kernel=3):
        super(ResNeXtDownSample, self).__init__()

        sub_net_layers = [nn.Conv2d(in_channels, in_channels // 2, 1),
                          nn.BatchNorm2d(in_channels // 2),
                          nn.LeakyReLU(),
                          nn.Conv2d(in_channels // 2, in_channels // 2, kernel, padding=1, groups=c),
                          nn.BatchNorm2d(in_channels // 2),
                          nn.LeakyReLU(),
                          nn.Conv2d(in_channels // 2, out_channels, 1),
                          nn.BatchNorm2d(out_channels),
                          nn.LeakyReLU()]

        self.path = nn.Sequential(*sub_net_layers)
        self.identity_path = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                           nn.BatchNorm2d(out_channels))

    def forward(self, x):
        path = self.path(x)
        id_path = self.identity_path(x)

        return F.leaky_relu(path + id_path)


class ResNeXtUpSample(nn.Module):
    """
    Wrong implementation
    """

    def __init__(self, in_channels, out_channels):
        super(ResNeXtUpSample, self).__init__()

        sub_net_layers = [nn.ConvTranspose2d(in_channels, out_channels, 1, 1),
                          nn.BatchNorm2d(out_channels),
                          nn.LeakyReLU(),
                          nn.ConvTranspose2d(out_channels, 4, 1, 1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.ConvTranspose2d(4, 4, 3, 1, 1),
                          nn.BatchNorm2d(4),
                          nn.LeakyReLU(),
                          nn.ConvTranspose2d(4, out_channels, 1, 1)]

        self.path = nn.Sequential(*sub_net_layers)
        self.identity_path = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 1, 1),
                                           nn.BatchNorm2d(out_channels))

    def forward(self, x):
        xs = [self.path(x) for _ in range(2)]
        early_agg = xs[0]

        for x_ in xs[1:]:
            early_agg += x_

        late_agg = F.leaky_relu(F.leaky_relu(early_agg) + self.identity_path(x))

        return late_agg


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


class ResidualSEDownSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualSEDownSample, self).__init__()

        self.res_path = ResidualDownSample(in_channels, out_channels)
        self.se_path = SEBlock(out_channels)
        self.identity_path = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                           nn.BatchNorm2d(out_channels))

    def forward(self, x):
        main_path = self.se_path(self.res_path(x))
        id_path = self.identity_path(x)

        return F.leaky_relu(main_path + id_path)


class ResidualSEUpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualSEUpSample, self).__init__()

        self.res_path = ResidualUpSample(in_channels, out_channels)
        self.se_path = SEBlock(out_channels)
        self.identity_path = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                           nn.BatchNorm2d(out_channels))

    def forward(self, x):
        main_path = self.se_path(self.res_path(x))
        id_path = self.identity_path(x)

        return F.leaky_relu(main_path + id_path)


class SplitAttention(nn.Module):

    def __init__(self, in_channels, out_channels, g=1, radix=2, red_fac=4):
        super(SplitAttention, self).__init__()

        self.radix = radix
        self.radix_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * radix, 3,
                                                  groups=g * radix, padding=1),
                                        nn.BatchNorm2d(out_channels * radix),
                                        nn.LeakyReLU(inplace=True))
        inter_channels = max(32, in_channels * radix // red_fac)
        self.attention = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, groups=g),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.Conv2d(inter_channels, out_channels * radix, 1, groups=g))
        self.r_softmax = RSoftMax(g, radix)

    def forward(self, x):
        x = self.radix_conv(x)
        size, rc = x.size()[:2]
        splits = torch.split(x, rc // self.radix, dim=1)
        gap = sum(splits)
        att_map = self.r_softmax(self.attention(gap))

        att_maps = torch.split(att_map, rc // self.radix, dim=1)
        out = sum([att_map * split for att_map, split in zip(att_maps, splits)])

        return out.contiguous()


class ResNeStBlock(nn.Module):

    def __init__(self, in_channels, out_channels, radix=2, g=1, bottleneck_width=64):
        super(ResNeStBlock, self).__init__()

        self.cardinality = 4
        self.grp_width = int(out_channels * (bottleneck_width / 64)) * g
        layers = [nn.Conv2d(in_channels, self.grp_width, 1),
                  SplitAttention(self.grp_width, self.grp_width, g, radix),
                  nn.Conv2d(self.grp_width, out_channels, 1),
                  nn.BatchNorm2d(out_channels)]

        self.block = nn.Sequential(*layers)
        self.id_path = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                     nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += self.id_path(x)

        return F.leaky_relu(out)


if __name__ == "__main__":
    t = torch.randn(4, 16, 48, 48)

    model = ResNeStBlock(16, 16)

    print(model(t).shape)
