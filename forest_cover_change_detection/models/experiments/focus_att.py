import torch
from torch import nn
from torch.nn import functional as F


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


class FocusAttentionGateV2(nn.Module):

    def __init__(self, gate_channels, skip_channels):
        super(FocusAttentionGateV2, self).__init__()
        self.query = nn.Conv2d(gate_channels, skip_channels, 1, device='cuda')
        self.value = nn.Conv2d(skip_channels, skip_channels, 1, device='cuda')
        self.resampler = nn.ConvTranspose2d(skip_channels, skip_channels, 3, padding=1, device='cuda')
        self.up_sample = nn.ConvTranspose2d(skip_channels, skip_channels, 3, 2, 1, 1, device='cuda')
        self.channel_att = ChannelAttention(skip_channels, skip_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, skip_con, gate_signal):  # (16, 128, 16, 16), (16, 256, 8, 8)
        gate = self.up_sample(self.query(gate_signal))  # (16, 128, 16, 16)
        skip = self.value(skip_con)  # (16, 128, 16, 16)
        merge = F.relu(skip + gate)  # (16, 128, 16, 16)

        ch_at = self.channel_att(merge)
        sp_at = self.spatial_att(merge)
        resample = self.resampler(torch.exp(ch_at * sp_at))

        return resample * skip


if __name__ == '__main__':
    s = torch.randn(4, 8, 64, 64).cuda()
    g = torch.randn(4, 16, 32, 32).cuda()
    m = FocusAttentionGateV2(16, 8)

    print(m(s, g).shape)
