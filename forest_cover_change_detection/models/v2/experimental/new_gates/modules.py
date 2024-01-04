import torch

from torch import nn
from torch.nn import functional as F
from forest_cover_change_detection.models.fcfe_with_att.modules import StripPooling


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
        self.se = SqueezeExcitation(skip_channels).cuda()

        if not additive:
            self.press = nn.Conv2d(2 * skip_channels, skip_channels, 1, device='cuda')

    def forward(self, skip_con, gate_signal):
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((g + s))
            w = self.se(additive)

        else:
            concat = F.relu(self.press(torch.cat((g, s), dim=1)))
            w = self.se(concat)

        return skip_con * w


class DualAttentionV2(nn.Module):

    def __init__(self, gate_channels, skip_channels, additive=True):
        super(DualAttentionV2, self).__init__()

        self.additive = additive
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.vit = Vit(skip_channels).cuda()

        if not additive:
            self.press = nn.Conv2d(2 * skip_channels, skip_channels, 1, device='cuda')

    def forward(self, skip_con, gate_signal):
        _, _, w1, h1 = skip_con.shape
        _, _, w2, h2 = gate_signal.shape
        scale_by = w1 // w2
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((F.interpolate(g, scale_factor=scale_by) + s))
            w = self.vit(F.interpolate(additive, scale_factor=1 / scale_by))
            w = F.softmax(F.interpolate(w, scale_factor=scale_by), dim=-1)

        else:
            concat = F.relu(self.press(torch.cat((F.interpolate(g, scale_factor=scale_by), s), dim=1)))
            w = self.vit(F.interpolate(concat, scale_factor=1 / scale_by))
            w = F.softmax(F.interpolate(w, scale_factor=scale_by), dim=-1)

        return skip_con * w


class DualAttentionV3(nn.Module):

    def __init__(self, gate_channels, skip_channels, additive=True, pool_size=None):
        super(DualAttentionV3, self).__init__()

        self.additive = additive
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.vit = Vit(skip_channels).cuda()
        self.se = SqueezeExcitation(skip_channels).cuda()

        if pool_size is not None:
            self.app_sp = True
            self.strip_pool = StripPooling(skip_channels, pool_size)
        else:
            self.app_sp = False

        if not additive:
            self.press = nn.Conv2d(2 * skip_channels, skip_channels, 1, device='cuda')

    def forward(self, skip_con, gate_signal):
        _, _, w1, h1 = skip_con.shape
        _, _, w2, h2 = gate_signal.shape
        scale_by = w1 // w2
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((F.interpolate(g, scale_factor=scale_by) + s))
            c_w = self.se(additive)
            s_w = F.interpolate(self.vit(F.interpolate(additive, scale_factor=1 / scale_by)), scale_factor=scale_by)
            w = F.sigmoid(s_w * c_w)

        else:
            concat = F.relu(self.press(torch.cat((F.interpolate(g, scale_factor=scale_by), s), dim=1)))
            c_w = self.se(concat)
            s_w = F.interpolate(self.vit(F.interpolate(concat, scale_factor=1 / scale_by)), scale_factor=scale_by)
            w = F.sigmoid(s_w * c_w)

        out = skip_con * w

        if self.app_sp:
            sp = self.strip_pool(skip_con)

            return torch.cat((out, sp), dim=1)

        return out


class DualAttentionV4(nn.Module):

    def __init__(self, gate_channels, skip_channels, additive=True, pool_size=None):
        super(DualAttentionV4, self).__init__()

        self.app_sp = False
        self.additive = additive
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.vit = Vit(skip_channels).cuda()
        self.se = SqueezeExcitation(skip_channels).cuda()

        if pool_size is not None:
            self.app_sp = True
            self.strip_pool = StripPooling(skip_channels, pool_size)

        if not additive:
            self.press = nn.Conv2d(2 * skip_channels, skip_channels, 1, device='cuda')

    def forward(self, skip_con, gate_signal):
        _, _, w1, h1 = skip_con.shape
        _, _, w2, h2 = gate_signal.shape
        scale_by = w1 // w2
        g = self.query(gate_signal)
        s = self.value(skip_con)

        if self.additive:
            additive = F.relu((F.interpolate(g, scale_factor=scale_by) + s))
            c_w = self.se(additive)
            s_w = F.interpolate(self.vit(F.interpolate(additive, scale_factor=1 / scale_by)), scale_factor=scale_by)

            out = torch.cat((skip_con * c_w, s_w), dim=1)

        else:
            concat = F.relu(self.press(torch.cat((F.interpolate(g, scale_factor=scale_by), s), dim=1)))
            c_w = self.se(concat)
            s_w = F.interpolate(self.vit(F.interpolate(concat, scale_factor=1 / scale_by)), scale_factor=scale_by)

            out = torch.cat((skip_con * c_w, s_w), dim=1)

        if self.app_sp:
            sp = self.strip_pool(skip_con)

            return torch.cat((out, sp), dim=1)

        return out


class DualAttentionV5(nn.Module):

    def __init__(self, gate_channels, skip_channels, pool_size, concat_ops='skip'):
        super(DualAttentionV5, self).__init__()

        self.concat_ops = concat_ops
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.beta = nn.Conv2d(skip_channels, 1, 1, device='cuda')
        self.se = SqueezeExcitation(skip_channels).cuda()
        self.strip_pool = StripPooling(skip_channels, pool_size)

    def forward(self, skip_con, gate_signal):
        _, _, w1, h1 = skip_con.shape
        _, _, w2, h2 = gate_signal.shape
        scale_by = w1 // w2

        g = self.query(gate_signal)
        s = self.value(skip_con)

        additive = F.relu((F.interpolate(g, scale_factor=scale_by) + s))
        s_w = F.sigmoid(self.beta(additive))
        c_w = F.relu(self.se(additive))
        co_w = self.strip_pool(additive)

        if self.concat_ops == 'skip':
            out = skip_con * s_w + skip_con * c_w + F.relu(skip_con + co_w)
        else:
            out = additive * s_w + additive * c_w + F.relu(additive + co_w)

        return out


if __name__ == '__main__':
    s = torch.randn(4, 32, 64, 64).cuda()
    g = torch.randn(4, 64, 32, 32).cuda()
    m = DualAttentionV5(64, 32, (40, 24)).cuda()

    print(m(s, g).shape)
