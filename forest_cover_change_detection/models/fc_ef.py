import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import UpSample, DownSample


class FCFE(nn.Module):

    def __init__(self, in_channels, kernel, classes):
        super(FCFE, self).__init__()

        self.config = [16, 32, 64, 128]
        self.max_pooling = nn.MaxPool2d(2)

        # sub-sampling blocks
        self.dwn_block1 = DownSample(in_channels, self.config[0], kernel, blocks=2)
        self.dwn_block2 = DownSample(self.config[0], self.config[1], kernel, blocks=2)
        self.dwn_block3 = DownSample(self.config[1], self.config[2], kernel, blocks=3)
        self.dwn_block4 = DownSample(self.config[2], self.config[3], kernel, blocks=3)

        # up-sampling blocks
        self.up_layer1 = UpSample(self.config[3], self.config[3], kernel, stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)
        self.up_block1 = nn.Sequential(UpSample(2 * self.config[3], self.config[3], kernel, padding=1, blocks=2),
                                       UpSample(self.config[3], self.config[2], kernel, padding=1, blocks=1)
                                       )

        self.up_layer2 = UpSample(self.config[2], self.config[2], kernel, stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)
        self.up_block2 = nn.Sequential(UpSample(2 * self.config[2], self.config[2], kernel, padding=1, blocks=2),
                                       UpSample(self.config[2], self.config[1], kernel, padding=1, blocks=1)
                                       )

        self.up_layer3 = UpSample(self.config[1], self.config[1], kernel, stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)
        self.up_block3 = nn.Sequential(UpSample(2 * self.config[1], self.config[1], kernel, padding=1, blocks=1),
                                       UpSample(self.config[1], self.config[0], kernel, padding=1, blocks=1)
                                       )

        self.up_layer4 = UpSample(self.config[0], self.config[0], kernel, stride=2, output_padding=1, blocks=1,
                                  batch_norm=False, dropout=False)
        self.up_block4 = nn.Sequential(
            UpSample(2 * self.config[0], self.config[0], kernel, padding=1, blocks=1),
            nn.ConvTranspose2d(self.config[0], classes, kernel, padding=1)
        )
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_1 = self.dwn_block1(x)
        x_1_m = self.max_pooling(x_1)
        x_2 = self.dwn_block2(x_1_m)
        x_2_m = self.max_pooling(x_2)
        x_3 = self.dwn_block3(x_2_m)
        x_3_m = self.max_pooling(x_3)
        x_4 = self.dwn_block4(x_3_m)
        x_5_m = self.max_pooling(x_4)

        x_6 = self.up_layer1(x_5_m)

        pad_x6 = ReplicationPad2d((0, x_4.size(3) - x_6.size(3), 0, x_4.size(2) - x_6.size(2)))
        x_6 = torch.cat((pad_x6(x_6), x_4), dim=1)

        x_7 = self.up_block1(x_6)
        x_8 = self.up_layer2(x_7)

        pad_x8 = ReplicationPad2d((0, x_3.size(3) - x_8.size(3), 0, x_3.size(2) - x_8.size(2)))
        x_8 = torch.cat((pad_x8(x_8), x_3), dim=1)

        x_9 = self.up_block2(x_8)
        x_10 = self.up_layer3(x_9)

        pad_x10 = ReplicationPad2d((0, x_2.size(3) - x_10.size(3), 0, x_2.size(2) - x_10.size(2)))
        x_10 = torch.cat((pad_x10(x_10), x_2), dim=1)

        x_11 = self.up_block3(x_10)
        x_12 = self.up_layer4(x_11)

        pad_x12 = ReplicationPad2d((0, x_1.size(3) - x_12.size(3), 0, x_1.size(2) - x_12.size(2)))
        x_12 = torch.cat((pad_x12(x_12), x_1), dim=1)

        x_13 = self.up_block4(x_12)

        return self.sm(x_13)


if __name__ == "__main__":
    t = torch.randn(16, 6, 128, 128)

    sub_sampling = FCFE(6, 3, classes=2)
    out = sub_sampling(t)

    print(out.shape)
