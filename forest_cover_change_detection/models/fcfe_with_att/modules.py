import math
import warnings
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from forest_cover_change_detection.layers.conv_ import AdaConv2d

warnings.filterwarnings(action='ignore')


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


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, n_heads=8, sr_ratio=1):
        super(EfficientAttention, self).__init__()

        head_dim = in_channels // n_heads
        self.scale = head_dim ** -0.5
        self.query = nn.Linear(in_channels, in_channels)
        self.key_value = nn.Linear(in_channels, in_channels * 2)
        self.proj = nn.Linear(in_channels, in_channels)
        self.n_heads = n_heads
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels, in_channels, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.query(x).view(b, n, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).view(b, c, h, w)
            x_ = self.sr(x_).view(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.key_value(x_).view(b, -1, 2, self.n_heads, c // self.n_heads).permute(2, 0, 3, 1, 4)

        else:
            kv = self.key_value(x).view(b, -1, 2, self.n_heads, c // self.n_heads).permute(2, 0, 3, 1, 4)

        key, value = kv[0], kv[1]
        attn = (q @ key.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ value).transpose(1, 2)  # .view(b, n, c)
        x = self.proj(x)

        return x


class StripPooling(nn.Module):

    def __init__(self, in_channels, pool_size):
        super(StripPooling, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        channels = in_channels // 4
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels),
                                     nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels),
                                     nn.ReLU())
        self.conv2_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(channels, channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels),
                                     nn.ReLU())
        self.conv2_6 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels),
                                     nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channels * 2, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))

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

        return F.leaky_relu(x + x_out)


class SelfCalibrationConvGate(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, dilation, groups, output_pooling):
        super(SelfCalibrationConvGate, self).__init__()

        self.k2 = nn.Sequential(nn.AvgPool2d(output_pooling, output_pooling),
                                nn.Conv2d(in_channels, out_channels, 3, 1,
                                          padding=padding, dilation=dilation, groups=groups, bias=False),
                                nn.BatchNorm2d(out_channels))
        self.k3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1,
                                          padding=padding, dilation=dilation, groups=groups, bias=False),
                                nn.BatchNorm2d(out_channels))
        self.k4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride,
                                          padding=padding, dilation=dilation, groups=groups, bias=False),
                                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.sigmoid(torch.add(x, F.interpolate(self.k2(x), x.shape[2:])))
        out = torch.multiply(self.k3(x), out)
        out = self.k4(out)

        return out


if __name__ == '__main__':
    s = torch.randn(16, 16, 128, 128).cuda()
    print(f"s: {s.shape}")

    model = SelfCalibrationConvGate(16, 16, 1,
                                    2, 2, 1, 4)
    model.cuda()

    print(model(s).shape)
