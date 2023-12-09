import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResidualDownSample, ResidualUpSample


class FCFEResDiff(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEResDiff, self).__init__()
        filters = [8, 16, 32, 64, 128]
        self.drop = nn.Dropout(0.2)

        # down sampling
        self.dwn_block_1 = nn.Sequential(ResidualDownSample(in_channels, filters[0]),
                                         ResidualDownSample(filters[0], filters[0])
                                         )
        self.dwn_block_2 = nn.Sequential(ResidualDownSample(filters[0], filters[1]),
                                         ResidualDownSample(filters[1], filters[1])
                                         )
        self.dwn_block_3 = nn.Sequential(ResidualDownSample(filters[1], filters[2]),
                                         ResidualDownSample(filters[2], filters[2])
                                         )
        self.dwn_block_4 = nn.Sequential(ResidualDownSample(filters[2], filters[3]),
                                         ResidualDownSample(filters[3], filters[3])
                                         )
        self.dwn_block_5 = nn.Sequential(ResidualDownSample(filters[3], filters[4]),
                                         ResidualDownSample(filters[4], filters[4])
                                         )

        # up sampling
        self.up_block_1 = nn.Sequential(ResidualDownSample(filters[4], filters[4]),
                                        ResidualUpSample(filters[4], 2 * filters[4])
                                        )
        self.up_block_2 = nn.Sequential(ResidualDownSample(3 * filters[4], filters[4]),
                                        ResidualUpSample(filters[4], filters[4])
                                        )
        self.up_block_3 = nn.Sequential(ResidualDownSample(3 * filters[3], filters[3]),
                                        ResidualUpSample(filters[3], filters[3])
                                        )
        self.up_block_4 = nn.Sequential(ResidualDownSample(3 * filters[2], filters[2]),
                                        ResidualUpSample(filters[2], filters[2])
                                        )
        self.up_block_5 = nn.Sequential(ResidualDownSample(3 * filters[1], filters[1]),
                                        ResidualUpSample(filters[1], filters[1])
                                        )
        self.out_block = nn.Sequential(ResidualDownSample(3 * filters[0], filters[0]),
                                       nn.Conv2d(filters[0], classes, kernel, padding=1),
                                       nn.LogSoftmax(dim=1)
                                       )

    def forward(self, x1, x2):
        s_0 = x1.shape
        s_1_ = x2.shape

        # branch 1
        xd_1 = self.dwn_block_1(x1)
        s_1 = xd_1.shape
        xd_2 = self.dwn_block_2(xd_1)
        s_2 = xd_2.shape
        xd_3 = self.dwn_block_3(xd_2)
        s_3 = xd_3.shape
        xd_4 = self.dwn_block_4(xd_3)
        s_4 = xd_4.shape
        xd_5 = self.dwn_block_5(xd_4)
        s_5 = xd_5.shape

        # branch 2
        xd_1_1 = self.dwn_block_1(x2)
        s_1_1 = xd_1_1.shape
        xd_2_1 = self.dwn_block_2(xd_1_1)
        s_2_1 = xd_2_1.shape
        xd_3_1 = self.dwn_block_3(xd_2_1)
        s_3_1 = xd_3_1.shape
        xd_4_1 = self.dwn_block_4(xd_3_1)
        s_4_1 = xd_4_1.shape
        xd_5_1 = self.dwn_block_5(xd_4_1)
        s_5_1 = xd_5_1.shape

        # decoder
        xu_1 = self.up_block_1(xd_5)
        pad_1 = ReplicationPad2d((0, (s_4[3] - s_5[3]), 0, (s_4[2] - s_5[2])))
        xu_1 = torch.cat((xu_1, pad_1(xd_5 - xd_5_1)), dim=1)
        xu_1 = self.drop(xu_1)

        xu_2 = self.up_block_2(xu_1)
        pad_2 = ReplicationPad2d((0, (s_3[3] - s_4[3]), 0, (s_3[2] - s_4[2])))
        xu_2 = torch.cat((xu_2, pad_2(xd_4 - xd_4_1)), dim=1)

        xu_3 = self.up_block_3(xu_2)
        pad_3 = ReplicationPad2d((0, (s_2[3] - s_3[3]), 0, (s_2[2] - s_3[2])))
        xu_3 = torch.cat((xu_3, pad_3(xd_3 - xd_3_1)), dim=1)

        xu_4 = self.up_block_4(xu_3)
        pad_4 = ReplicationPad2d((0, (s_1[3] - s_2[3]), 0, (s_1[2] - s_2[2])))
        xu_4 = torch.cat((xu_4, pad_4(xd_2 - xd_2_1)), dim=1)

        xu_5 = self.up_block_5(xu_4)
        pad_5 = ReplicationPad2d((0, (s_0[3] - s_1[3]), 0, (s_0[2] - s_1[2])))
        xu_5 = torch.cat((xu_5, pad_5(xd_1 - xd_1_1)), dim=1)

        x_out = self.out_block(xu_5)

        return x_out


if __name__ == '__main__':
    t = torch.randn(16, 6, 256, 256)
    model = FCFEResDiff(6, 2)

    print(model(t, t).shape)
