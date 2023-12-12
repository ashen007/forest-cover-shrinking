import math

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F


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


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(2 * in_channels, out_channels, 1, device='cuda')

    def forward(self, x):
        glob_avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        glob_max_pool = F.max_pool2d(x, x.shape[2:])

        return F.sigmoid(self.conv(torch.cat((glob_avg_pool, glob_max_pool), dim=1)))


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 1, device='cuda')

    def forward(self, x):
        glob_avg_pool = torch.mean(x, dim=1, keepdim=True)
        glob_max_pool = torch.amax(x, dim=1, keepdim=True)

        return F.sigmoid(self.conv(torch.cat((glob_avg_pool, glob_max_pool), dim=1)))


class FocusAttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, stride, padding, out_padding):
        super(FocusAttentionGate, self).__init__()
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.up_sample = nn.ConvTranspose2d(skip_channels, skip_channels, 3, stride, padding, out_padding,
                                            device='cuda')
        self.resampler = nn.ConvTranspose2d(skip_channels, skip_channels, 3, padding=1, device='cuda')
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


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):

    def __init__(self, h, w, map_x, map_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        self.num_freq = len(map_x)
        self.register_buffer('weight', self.get_dct_filter(h, w, map_x, map_y, channel))

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)

        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

    def forward(self, x):
        x = x * self.weight
        result = torch.sum(x, dim=(2, 3))

        return result


class MultiSpectralAttentionLayer(nn.Module):

    def __init__(self, in_channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super(MultiSpectralAttentionLayer, self).__init__()

        self.red = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        map_x, map_y = get_freq_indices(freq_sel_method)
        self.num_split = len(map_x)

        map_x = [temp_x * (dct_h // 7) for temp_x in map_x]
        map_y = [temp_y * (dct_w // 7) for temp_y in map_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, map_x, map_y, in_channel)
        self.fc = nn.Sequential(nn.Linear(in_channel, in_channel // reduction),
                                nn.LeakyReLU(),
                                nn.Linear(in_channel // reduction, in_channel),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x

        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)

        return x * y.expand_as(x)


if __name__ == '__main__':
    s = torch.randn(16, 32, 128, 128)

    model = MultiSpectralAttentionLayer(32, 14, 14)

    print(model(s).shape)
