import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResidualDownSample, ResidualUpSample


class FCFEResNeXt(nn.Module):

    def __init__(self, in_channels, classes):
        super(FCFEResNeXt, self).__init__()

    def forward(self, x):
        return
