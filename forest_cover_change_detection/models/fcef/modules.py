import torch

from torch import nn
from torch.nn import functional as F


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

    def __init__(self, in_channels):
        super(SplitAttention, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.main_block = nn.Sequential(nn.Linear(in_channels, in_channels),
                                        nn.BatchNorm1d(in_channels),
                                        nn.LeakyReLU())
        self.linear = nn.Linear(in_channels, in_channels)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x[0].shape
        ins = torch.zeros(x[0].shape).cuda()
        output = torch.zeros(x[0].shape).cuda()

        for i in x:
            ins += i

        x_ = self.pool(ins).view(b, c)
        x_ = self.main_block(x_)
        scores = [self.sm(self.linear(x_)).view(b, c, 1, 1) for _ in range(len(x))]

        for i, s in zip(x, scores):
            output += (i * s)

        return output


class ResNeStBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cardinality=1, splits=3):
        super(ResNeStBlock, self).__init__()

        self.split = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU()
                                   )
        if cardinality > 1:
            self.cardinal = [[self.split for _ in range(splits)] for _ in range(cardinality)]

        else:
            self.cardinal = [self.split for _ in range(splits)]

        self.split_attention = SplitAttention(out_channels)
        self.out = nn.Conv2d(cardinality * out_channels, out_channels, 1)
        self.identity_path = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                           nn.BatchNorm2d(out_channels))
        self.cardinality = cardinality

    def forward(self, x):
        if self.cardinality > 1:
            xs = []

            for i in range(self.cardinality):
                ins = [card(x) for card in self.cardinal[i]]
                xs.append(self.split_attention(ins))

            return self.out(torch.cat(xs, dim=1)) + self.identity_path(x)

        else:
            ins = [card(x) for card in self.cardinal]

            return self.out(self.split_attention(ins)) + self.identity_path(x)


if __name__ == "__main__":
    t = torch.randn(4, 16, 48, 48)

    model = ResNeXtDownSample(16, 32, 8)

    print(model(t).shape)
