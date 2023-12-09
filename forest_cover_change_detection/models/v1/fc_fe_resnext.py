import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResNeXtDownSample, ResNeXtUpSample


class FCFEResNeXt(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEResNeXt, self).__init__()
        filters = [16, 32, 64, 128, 256]
        self.drop = nn.Dropout(0.2)

        # down sampling
        self.dwn_block_1 = nn.Sequential(ResNeXtDownSample(in_channels, filters[0]),
                                         ResNeXtDownSample(filters[0], filters[0]),
                                         nn.MaxPool2d(2)  # (16, 128, 128)
                                         )
        self.dwn_block_2 = nn.Sequential(ResNeXtDownSample(filters[0], filters[1]),
                                         ResNeXtDownSample(filters[1], filters[1]),
                                         nn.MaxPool2d(2)  # (32, 64, 64)
                                         )
        self.dwn_block_3 = nn.Sequential(ResNeXtDownSample(filters[1], filters[2]),
                                         ResNeXtDownSample(filters[2], filters[2]),
                                         nn.MaxPool2d(2)  # (64, 32, 32)
                                         )
        self.dwn_block_4 = nn.Sequential(ResNeXtDownSample(filters[2], filters[3]),
                                         ResNeXtDownSample(filters[3], filters[3]),
                                         nn.MaxPool2d(2)  # (128, 16, 16)
                                         )
        self.dwn_block_5 = nn.Sequential(ResNeXtDownSample(filters[3], filters[4]),
                                         ResNeXtDownSample(filters[4], filters[4]),
                                         nn.MaxPool2d(2)  # (256, 8, 8)
                                         )

        # up sampling
        self.up_block_1 = nn.Sequential(ResNeXtUpSample(filters[4], filters[4]),
                                        nn.ConvTranspose2d(filters[4], filters[4], kernel,
                                                           stride=2, padding=0, output_padding=1)
                                        )  # (256, 16, 16)
        self.up_block_2 = nn.Sequential(ResNeXtUpSample(3 * filters[3], filters[4]),
                                        ResNeXtUpSample(filters[4], filters[3]),
                                        nn.ConvTranspose2d(filters[3], filters[3], kernel,
                                                           stride=2, padding=0, output_padding=1)
                                        )  # (128, 32, 32)
        self.up_block_3 = nn.Sequential(ResNeXtUpSample(3 * filters[2], filters[3]),
                                        ResNeXtUpSample(filters[3], filters[2]),
                                        nn.ConvTranspose2d(filters[2], filters[2], kernel,
                                                           stride=2, padding=0, output_padding=1)
                                        )  # (64, 64, 64)
        self.up_block_4 = nn.Sequential(ResNeXtUpSample(3 * filters[1], filters[2]),
                                        ResNeXtUpSample(filters[2], filters[1]),
                                        nn.ConvTranspose2d(filters[1], filters[1], kernel,
                                                           stride=2, padding=0, output_padding=1)
                                        )  # (32, 128, 128)
        self.up_block_5 = nn.Sequential(ResNeXtUpSample(3 * filters[0], filters[1]),
                                        ResNeXtUpSample(filters[1], filters[0]),
                                        nn.ConvTranspose2d(filters[0], filters[0], kernel,
                                                           stride=2, padding=0, output_padding=1)
                                        )  # (16, 256, 256)
        self.out_block = nn.Sequential(ResNeXtUpSample(filters[0], filters[0]),
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
        xd_5 = self.drop(xd_5)

        xu_1 = self.up_block_1(xd_5)
        pad_1 = ReplicationPad2d((0, (s_4[3] - xu_1.shape[3]), 0, (s_4[2] - xu_1.shape[2])))

        xu_1 = torch.cat((xd_4, pad_1(xu_1)), dim=1)
        xu_1 = self.drop(xu_1)

        xu_2 = self.up_block_2(xu_1)
        pad_2 = ReplicationPad2d((0, (s_3[3] - xu_2.shape[3]), 0, (s_3[2] - xu_2.shape[2])))

        xu_2 = torch.cat((xd_3, pad_2(xu_2)), dim=1)

        xu_3 = self.up_block_3(xu_2)
        pad_3 = ReplicationPad2d((0, (s_2[3] - xu_3.shape[3]), 0, (s_2[2] - xu_3.shape[2])))

        xu_3 = torch.cat((xd_2, pad_3(xu_3)), dim=1)

        xu_4 = self.up_block_4(xu_3)
        pad_4 = ReplicationPad2d((0, (s_1[3] - xu_4.shape[3]), 0, (s_1[2] - xu_4.shape[2])))

        xu_4 = torch.cat((xd_1, pad_4(xu_4)), dim=1)

        xu_5 = self.up_block_5(xu_4)
        pad_5 = ReplicationPad2d((0, (s_0[3] - xu_5.shape[3]), 0, (s_0[2] - xu_5.shape[2])))

        x_out = self.out_block(pad_5(xu_5))

        return x_out


if __name__ == '__main__':
    t = torch.randn(4, 6, 256, 256)
    model = FCFEResNeXt(6, 2)

    print(f'out: {model(t).shape}')
