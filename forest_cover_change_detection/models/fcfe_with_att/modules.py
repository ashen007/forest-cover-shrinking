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
