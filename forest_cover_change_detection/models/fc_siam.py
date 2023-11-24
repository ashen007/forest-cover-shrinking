import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import UpSample, DownSample


class FCSiam(nn.Module):

    def __init__(self, in_channels, classes, diff, kernel=3):
        super(FCSiam, self).__init__()

        self.config = [16, 32, 64, 128]
        self.max_pooling = nn.MaxPool2d(2)
        self.diff = diff

        # sub-sampling blocks
        self.dwn_block1 = DownSample(in_channels, self.config[0], kernel, blocks=2)
        self.dwn_block2 = DownSample(self.config[0], self.config[1], kernel, blocks=2)
        self.dwn_block3 = DownSample(self.config[1], self.config[2], kernel, blocks=3)
        self.dwn_block4 = DownSample(self.config[2], self.config[3], kernel, blocks=3)

        # up-sampling blocks
        self.up_layer1 = UpSample(self.config[3], self.config[3], kernel,
                                  stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)

        if self.diff:
            self.block_concat1 = UpSample(2 * self.config[3], self.config[3], kernel, padding=1, blocks=1)
        else:
            self.block_concat1 = UpSample(3 * self.config[3], self.config[3], kernel, padding=1, blocks=1)

        self.up_block1 = nn.Sequential(self.block_concat1,
                                       UpSample(self.config[3], self.config[3], kernel, padding=1, blocks=2),
                                       UpSample(self.config[3], self.config[2], kernel, padding=1, blocks=1)
                                       )

        self.up_layer2 = UpSample(self.config[2], self.config[2], kernel,
                                  stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)

        if self.diff:
            self.block_concat2 = UpSample(2 * self.config[2], self.config[2], kernel, padding=1, blocks=1)
        else:
            self.block_concat2 = UpSample(3 * self.config[2], self.config[2], kernel, padding=1, blocks=1)

        self.up_block2 = nn.Sequential(self.block_concat2,
                                       UpSample(self.config[2], self.config[2], kernel, padding=1, blocks=2),
                                       UpSample(self.config[2], self.config[1], kernel, padding=1, blocks=1)
                                       )

        self.up_layer3 = UpSample(self.config[1], self.config[1], kernel,
                                  stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)

        if self.diff:
            self.block_concat3 = UpSample(2 * self.config[1], self.config[1], kernel, padding=1, blocks=1)
        else:
            self.block_concat3 = UpSample(3 * self.config[1], self.config[1], kernel, padding=1, blocks=1)

        self.up_block3 = nn.Sequential(self.block_concat3,
                                       UpSample(self.config[1], self.config[0], kernel, padding=1, blocks=1)
                                       )

        self.up_layer4 = UpSample(self.config[0], self.config[0], kernel,
                                  stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)

        if self.diff:
            self.block_concat4 = UpSample(2 * self.config[0], self.config[0], kernel, padding=1, blocks=1)
        else:
            self.block_concat4 = UpSample(3 * self.config[0], self.config[0], kernel, padding=1, blocks=1)

        self.up_block4 = nn.Sequential(self.block_concat4,
                                       nn.ConvTranspose2d(self.config[0], classes, kernel, padding=1)
                                       )
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        # encoder branch one
        x_11 = self.dwn_block1(x1)
        x_11_m = self.max_pooling(x_11)
        x_21 = self.dwn_block2(x_11_m)
        x_21_m = self.max_pooling(x_21)
        x_31 = self.dwn_block3(x_21_m)
        x_31_m = self.max_pooling(x_31)
        x_41 = self.dwn_block4(x_31_m)
        x_51_m = self.max_pooling(x_41)

        # encoder branch two
        x_12 = self.dwn_block1(x2)
        x_12_m = self.max_pooling(x_12)
        x_22 = self.dwn_block2(x_12_m)
        x_22_m = self.max_pooling(x_22)
        x_32 = self.dwn_block3(x_22_m)
        x_32_m = self.max_pooling(x_32)
        x_42 = self.dwn_block4(x_32_m)

        # decoder
        x_0 = self.up_layer1(x_51_m)
        pad_x0 = ReplicationPad2d((0, x_41.size(3) - x_0.size(3), 0, x_41.size(2) - x_0.size(2)))

        if self.diff:
            x_0 = torch.cat((pad_x0(x_0), (x_41 - x_42)), dim=1)
        else:
            x_0 = torch.cat((pad_x0(x_0), x_41, x_42), dim=1)

        x_01 = self.up_block1(x_0)
        x_02 = self.up_layer2(x_01)
        pad_x1 = ReplicationPad2d((0, x_31.size(3) - x_02.size(3), 0, x_31.size(2) - x_02.size(2)))

        if self.diff:
            x_02 = torch.cat((pad_x1(x_02), (x_31 - x_32)), dim=1)
        else:
            x_02 = torch.cat((pad_x1(x_02), x_31, x_32), dim=1)

        x_03 = self.up_block2(x_02)
        x_04 = self.up_layer3(x_03)
        pad_x2 = ReplicationPad2d((0, x_21.size(3) - x_04.size(3), 0, x_21.size(2) - x_04.size(2)))

        if self.diff:
            x_04 = torch.cat((pad_x2(x_04), (x_21 - x_22)), dim=1)

        else:
            x_04 = torch.cat((pad_x2(x_04), x_21, x_22), dim=1)

        x_05 = self.up_block3(x_04)
        x_06 = self.up_layer4(x_05)
        pad_x3 = ReplicationPad2d((0, x_11.size(3) - x_06.size(3), 0, x_11.size(2) - x_06.size(2)))

        if self.diff:
            x_06 = torch.cat((pad_x3(x_06), (x_11 - x_12)), dim=1)
        else:
            x_06 = torch.cat((pad_x3(x_06), x_11, x_12), dim=1)

        x_13_ = self.up_block4(x_06)

        return self.sm(x_13_)


if __name__ == "__main__":
    t = torch.randn(16, 6, 128, 128)

    sub_sampling = FCSiam(6, 2, False)
    out = sub_sampling(t, t)

    print(out.shape)
