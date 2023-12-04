import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResidualDownSample, ResidualUpSample
from forest_cover_change_detection.models.fcfe_with_att.modules import AdditiveAttentionGate


class FCFEResAAt(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEResAAt, self).__init__()
        filters = [16, 32, 64, 128, 256]
        self.drop = nn.Dropout(0.2)

        # down sampling
        self.dwn_block_1 = nn.Sequential(ResidualDownSample(in_channels, filters[0]),
                                         ResidualDownSample(filters[0], filters[0]),
                                         nn.MaxPool2d(2)  # (16, 128, 128)
                                         )
        self.dwn_block_2 = nn.Sequential(ResidualDownSample(filters[0], filters[1]),
                                         ResidualDownSample(filters[1], filters[1]),
                                         nn.MaxPool2d(2)  # (32, 64, 64)
                                         )
        self.dwn_block_3 = nn.Sequential(ResidualDownSample(filters[1], filters[2]),
                                         ResidualDownSample(filters[2], filters[2]),
                                         nn.MaxPool2d(2)  # (64, 32, 32)
                                         )
        self.dwn_block_4 = nn.Sequential(ResidualDownSample(filters[2], filters[3]),
                                         ResidualDownSample(filters[3], filters[3]),
                                         nn.MaxPool2d(2)  # (128, 16, 16)
                                         )
        self.dwn_block_5 = nn.Sequential(ResidualDownSample(filters[3], filters[4]),
                                         ResidualDownSample(filters[4], filters[4]),
                                         nn.MaxPool2d(2)  # (256, 8, 8)
                                         )

        # up sampling
        self.con_block_1 = nn.Sequential(ResidualUpSample(filters[4], filters[4]), )
        self.up_block_1 = nn.ConvTranspose2d(filters[4], filters[4], kernel,
                                             stride=2, padding=0, output_padding=1)  # (256, 16, 16)

        self.con_block_2 = nn.Sequential(ResidualUpSample(3 * filters[3], filters[4]),
                                         ResidualUpSample(filters[4], filters[3]), )
        self.up_block_2 = nn.ConvTranspose2d(filters[3], filters[3], kernel,
                                             stride=2, padding=0, output_padding=1)  # (128, 32, 32)

        self.con_block_3 = nn.Sequential(ResidualUpSample(3 * filters[2], filters[3]),
                                         ResidualUpSample(filters[3], filters[2]), )
        self.up_block_3 = nn.ConvTranspose2d(filters[2], filters[2], kernel,
                                             stride=2, padding=0, output_padding=1)  # (64, 64, 64)

        self.con_block_4 = nn.Sequential(ResidualUpSample(3 * filters[1], filters[2]),
                                         ResidualUpSample(filters[2], filters[1]), )
        self.up_block_4 = nn.ConvTranspose2d(filters[1], filters[1], kernel,
                                             stride=2, padding=0, output_padding=1)  # (32, 128, 128)

        self.con_block_5 = nn.Sequential(ResidualUpSample(3 * filters[0], filters[1]),
                                         ResidualUpSample(filters[1], filters[0]), )
        self.up_block_5 = nn.ConvTranspose2d(filters[0], filters[0], kernel,
                                             stride=2, padding=0, output_padding=1)  # (16, 256, 256)

        self.out_block = nn.Sequential(ResidualUpSample(filters[0], filters[0]),
                                       nn.Conv2d(filters[0], classes, kernel, padding=1),
                                       nn.LogSoftmax(dim=1)
                                       )  # (2, 256, 256)

    def forward(self, x):
        s_0 = x.shape

        xd_1 = self.dwn_block_1(x)
        s_1 = xd_1.shape

        xd_2 = self.dwn_block_2(xd_1)
        s_2 = xd_2.shape

        xd_3 = self.dwn_block_3(xd_2)
        s_3 = xd_3.shape

        xd_4 = self.dwn_block_4(xd_3)
        s_4 = xd_4.shape

        xd_5 = self.dwn_block_5(xd_4)
        s_5 = xd_5.shape

        xu_1_b = self.con_block_1(xd_5)  # (256, 8, 8)
        xu_1 = self.up_block_1(xu_1_b)
        pad_1 = ReplicationPad2d((0, (s_4[3] - xu_1.shape[3]), 0, (s_4[2] - xu_1.shape[2])))
        at_gate_1 = AdditiveAttentionGate(xu_1_b.shape[1], s_4[1])
        xu_1 = torch.cat((at_gate_1(xd_4, xu_1_b), pad_1(xu_1)), dim=1)

        xu_2_b = self.con_block_2(xu_1)  # (128, 16, 16)
        xu_2 = self.up_block_2(xu_2_b)
        pad_2 = ReplicationPad2d((0, (s_3[3] - xu_2.shape[3]), 0, (s_3[2] - xu_2.shape[2])))
        at_gate_2 = AdditiveAttentionGate(xu_2_b.shape[1], s_3[1])
        xu_2 = torch.cat((at_gate_2(xd_3, xu_2_b), pad_2(xu_2)), dim=1)

        xu_3_b = self.con_block_3(xu_2)  # (64, 32, 32)
        xu_3 = self.up_block_3(xu_3_b)
        pad_3 = ReplicationPad2d((0, (s_2[3] - xu_3.shape[3]), 0, (s_2[2] - xu_3.shape[2])))
        at_gate_3 = AdditiveAttentionGate(xu_3_b.shape[1], s_2[1])
        xu_3 = torch.cat((at_gate_3(xd_2, xu_3_b), pad_3(xu_3)), dim=1)

        xu_4_b = self.con_block_4(xu_3)  # (32, 64, 64)
        xu_4 = self.up_block_4(xu_4_b)
        pad_4 = ReplicationPad2d((0, (s_1[3] - xu_4.shape[3]), 0, (s_1[2] - xu_4.shape[2])))
        at_gate_4 = AdditiveAttentionGate(xu_4_b.shape[1], s_1[1])
        xu_4 = torch.cat((at_gate_4(xd_1, xu_4_b), pad_4(xu_4)), dim=1)

        xu_5_b = self.con_block_5(xu_4)  # (16, 128, 128)
        xu_5 = self.up_block_5(xu_5_b)
        pad_5 = ReplicationPad2d((0, (s_0[3] - xu_5.shape[3]), 0, (s_0[2] - xu_5.shape[2])))

        x_out = self.out_block(pad_5(xu_5))

        return x_out


if __name__ == '__main__':
    t = torch.randn(16, 6, 256, 256)
    model = FCFEResAAt(6, 2)

    print(model(t).shape)
