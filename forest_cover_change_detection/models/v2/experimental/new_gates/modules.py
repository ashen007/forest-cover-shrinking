import torch

from torch import nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reducer=4):
        super(SqueezeExcitation, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(nn.Linear(in_channels, in_channels // reducer),
                                      nn.LeakyReLU(),
                                      nn.Linear(in_channels // reducer, in_channels),
                                      nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        x_ = self.pool(x).view(b, c)

        x_ = self.se_block(x_).view(b, c, 1, 1)

        return x_


class StripPooling(nn.Module):

    def __init__(self, in_channels, pool_size):
        super(StripPooling, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        channels = in_channels // 4
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'),
                                     nn.ReLU())
        self.conv2_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'))
        self.conv2_4 = nn.Sequential(nn.Conv2d(channels, channels, (3, 1), 1, (1, 0), bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'))
        self.conv2_5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'),
                                     nn.ReLU())
        self.conv2_6 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False, device='cuda'),
                                     nn.BatchNorm2d(channels, device='cuda'),
                                     nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channels * 2, in_channels, 1, bias=False, device='cuda'),
                                   nn.BatchNorm2d(in_channels, device='cuda'))

    def forward(self, x):
        _, _, h, w = x.shape

        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))

        x1 = self.conv2_5(F.leaky_relu(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.leaky_relu(x2_5 + x2_4))
        x_out = self.conv3(torch.cat([x1, x2], dim=1))

        return x_out


class Vit(nn.Module):

    def __init__(self, in_channels):
        super(Vit, self).__init__()

        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        n, c, w, h = x.shape
        proj_q = self.query(x).view(n, -1, w * h).permute(0, 2, 1)
        proj_k = self.key(x).view(n, -1, w * h)
        energy = torch.bmm(proj_q, proj_k)
        att_scores = F.softmax(energy, dim=-1)
        proj_v = self.value(x).view(n, -1, w * h)

        return torch.bmm(proj_v, att_scores.permute(0, 2, 1)).view(n, -1, w, h)


class DualAttentionV1(nn.Module):

    def __init__(self, gate_channels, skip_channels, additive=True):
        super(DualAttentionV1, self).__init__()

        self.additive = additive
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, 2, device='cuda')

        if additive:
            self.se = SqueezeExcitation(skip_channels).cuda()
        else:
            self.se = SqueezeExcitation(2 * skip_channels).cuda()

    def forward(self, skip_con, gate_signal):
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((g + s))
            w = F.sigmoid(self.se(additive))

        else:
            concat = F.relu(torch.cat((g, s), dim=1))
            w = F.sigmoid(self.se(concat))

        return skip_con * w


class DualAttentionV2(nn.Module):

    def __init__(self, gate_channels, skip_channels, additive=True):
        super(DualAttentionV2, self).__init__()

        self.additive = additive
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')

        if additive:
            self.vit = Vit(skip_channels).cuda()
        else:
            self.vit = Vit(2 * skip_channels).cuda()

    def forward(self, skip_con, gate_signal):
        _, _, w1, h1 = skip_con.shape
        _, _, w2, h2 = gate_signal.shape
        scale_by = w1 // w2
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((F.interpolate(g, scale_factor=scale_by) + s))
            w = F.sigmoid(self.vit(additive))

        else:
            concat = F.relu(torch.cat((F.interpolate(g, scale_factor=scale_by), s), dim=1))
            w = F.sigmoid(self.vit(concat))

        return skip_con * w


if __name__ == '__main__':
    s = torch.randn(4, 16, 16, 16).cuda()
    g = torch.randn(4, 32, 4, 4).cuda()
    m = DualAttentionV2(32, 16).cuda()

    print(m(s, g).shape)
