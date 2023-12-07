import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from forest_cover_change_detection.layers.conv_ import AdaConv2d


class SelfAttention(nn.Module):

    def __init__(self, channels):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.key = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.value = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x):
        size = x.shape
        x = x.view(*size[:2], -1)  # (16, 128, 256)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g) / np.sqrt(self.channels), dim=1)  # (16, 256, 256)
        out = torch.bmm(h, beta)  # (16, 128, 256)

        return out.view(*size).contiguous()


class AdditiveAttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels):
        super(AdditiveAttentionGate, self).__init__()

        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, 2, device='cuda')
        self.beta = nn.Conv2d(skip_channels, 1, 1, device='cuda')

    def forward(self, skip_con, gate_signal):
        g = self.query(gate_signal)
        s = self.value(skip_con)

        h = F.relu((g + s))
        w = F.sigmoid(self.beta(h))
        w = F.interpolate(w, scale_factor=2)

        return skip_con * w


class SqueezeAndExpand(nn.Module):

    def __init__(self, in_channels, embeddings):
        super(SqueezeAndExpand, self).__init__()

        self.inc_block = nn.Conv2d(in_channels, embeddings, 1)
        self.dec_block = nn.Conv2d(embeddings, in_channels, 1)

    def forward(self, x, score):
        return self.dec_block(self.inc_block(x) * score)


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(2 * in_channels, out_channels, 1)

    def forward(self, x):
        glob_avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        glob_max_pool = F.max_pool2d(x, x.shape[2:])

        return F.sigmoid(self.conv(torch.cat((glob_avg_pool, glob_max_pool), dim=1)))


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        glob_avg_pool = torch.mean(x, dim=1, keepdim=True)
        glob_max_pool = torch.amax(x, dim=1, keepdim=True)

        return F.sigmoid(self.conv(torch.cat((glob_avg_pool, glob_max_pool), dim=1)))


class FocusAttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, stride, padding, out_padding):
        super(FocusAttentionGate, self).__init__()
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.up_sample = nn.ConvTranspose2d(skip_channels, skip_channels, 3, stride, padding, out_padding)
        self.resampler = nn.ConvTranspose2d(skip_channels, skip_channels, 3, padding=1)
        self.channel_att = ChannelAttention(skip_channels, skip_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, skip_con, gate_signal):  # (16, 128, 16, 16), (16, 256, 8, 8)
        gate = self.up_sample(self.query(gate_signal))  # (16, 128, 16, 16)
        skip = self.value(skip_con)  # (16, 128, 16, 16)
        merge = F.relu(skip + gate)  # (16, 128, 16, 16)

        ch_at = self.channel_att(merge)
        sp_at = self.spatial_att(merge)
        resamp = self.resampler(torch.exp(ch_at * sp_at))

        return resamp * skip


if __name__ == '__main__':
    s = torch.randn(16, 128, 128, 128).cuda()
    g = torch.randn(16, 256, 8, 8).cuda()

    at = FocusAttentionGate(256, 128, 16, 1, 15)
    at.cuda()
    print(at(s, g).shape)
