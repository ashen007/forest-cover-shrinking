import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResNeXtDownSample, ResNeXtUpSample


class FCFEResNeXt(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEResNeXt, self).__init__()
        filters = [8, 16, 32, 64, 128]
        self.drop = nn.Dropout(0.2)

        # down sampling
        self.dwn_block_1 = nn.Sequential(nn.Conv2d(in_channels, filters[0], 7, padding=3),
                                         nn.BatchNorm2d(filters[0]),
                                         nn.LeakyReLU(),
                                         ResNeXtDownSample(filters[0], filters[0]),
                                         ResNeXtDownSample(filters[0], filters[0], down_sample=True)
                                         )
        self.dwn_block_2 = nn.Sequential(ResNeXtDownSample(filters[0], filters[1]),
                                         ResNeXtDownSample(filters[1], filters[1], down_sample=True)
                                         )
        self.dwn_block_3 = nn.Sequential(ResNeXtDownSample(filters[1], filters[2]),
                                         ResNeXtDownSample(filters[2], filters[2], down_sample=True)
                                         )
        self.dwn_block_4 = nn.Sequential(ResNeXtDownSample(filters[2], filters[3]),
                                         ResNeXtDownSample(filters[3], filters[3], down_sample=True)
                                         )
        self.dwn_block_5 = nn.Sequential(ResNeXtDownSample(filters[3], filters[4]),
                                         ResNeXtDownSample(filters[4], filters[4], down_sample=True),
                                         nn.Dropout(0.1)
                                         )

        # up sampling
        self.up_block_1 = nn.Sequential(ResNeXtDownSample(filters[4], filters[4]),
                                        ResNeXtUpSample(filters[4], 2 * filters[4])
                                        )
        self.up_block_2 = nn.Sequential(ResNeXtDownSample(3 * filters[4], filters[4]),
                                        ResNeXtUpSample(filters[4], filters[4]),
                                        )
        self.up_block_3 = nn.Sequential(ResNeXtDownSample(3 * filters[3], filters[3]),
                                        ResNeXtUpSample(filters[3], filters[3])
                                        )
        self.up_block_4 = nn.Sequential(ResNeXtDownSample(3 * filters[2], filters[2]),
                                        ResNeXtUpSample(filters[2], filters[2])
                                        )
        self.up_block_5 = nn.Sequential(ResNeXtDownSample(3 * filters[1], filters[1]),
                                        ResNeXtUpSample(filters[1], filters[1])
                                        )
        self.out_block = nn.Sequential(ResNeXtDownSample(3 * filters[0], filters[0]),
                                       nn.Conv2d(filters[0], classes, kernel, padding=1),
                                       nn.LogSoftmax(dim=1)
                                       )

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

        xu_1 = self.up_block_1(xd_5)
        pad_1 = ReplicationPad2d((0, (s_4[3] - s_5[3]), 0, (s_4[2] - s_5[2])))
        xu_1 = torch.cat((xu_1, pad_1(xd_5)), dim=1)

        xu_2 = self.up_block_2(xu_1)
        pad_2 = ReplicationPad2d((0, (s_3[3] - s_4[3]), 0, (s_3[2] - s_4[2])))
        xu_2 = torch.cat((xu_2, pad_2(xd_4)), dim=1)

        xu_3 = self.up_block_3(xu_2)
        pad_3 = ReplicationPad2d((0, (s_2[3] - s_3[3]), 0, (s_2[2] - s_3[2])))
        xu_3 = torch.cat((xu_3, pad_3(xd_3)), dim=1)

        xu_4 = self.up_block_4(xu_3)
        pad_4 = ReplicationPad2d((0, (s_1[3] - s_2[3]), 0, (s_1[2] - s_2[2])))
        xu_4 = torch.cat((xu_4, pad_4(xd_2)), dim=1)

        xu_5 = self.up_block_5(xu_4)
        pad_5 = ReplicationPad2d((0, (s_0[3] - s_1[3]), 0, (s_0[2] - s_1[2])))
        xu_5 = torch.cat((xu_5, pad_5(xd_1)), dim=1)

        x_out = self.out_block(xu_5)

        return x_out


if __name__ == '__main__':
    t = torch.randn(4, 6, 256, 256)
    model = FCFEResNeXt(6, 2)

    print(f'out: {model(t).shape}')
