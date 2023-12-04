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


class SqueezeAndExpand(nn.Module):

    def __init__(self, in_channels, embeddings):
        super(SqueezeAndExpand, self).__init__()

        self.inc_block = nn.Conv2d(in_channels, embeddings, 1)
        self.dec_block = nn.Conv2d(embeddings, in_channels, 1)

    def forward(self, x, score):
        return self.dec_block(self.inc_block(x) * score)


if __name__ == '__main__':
    s = torch.randn(16, 128, 16, 16)
    g = torch.randn(16, 256, 8, 8)

    at = AdditiveAttentionGate(256, 128)
    print(at(s, g).shape)
