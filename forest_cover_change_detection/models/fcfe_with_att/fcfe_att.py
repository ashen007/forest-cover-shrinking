import numpy as np
import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from torch.nn import functional as F
from .modules import SelfAttention


class FCFEWithAttention(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEWithAttention, self).__init__()

        self.config = [16, 32, 64, 128]
        self.self_attention = SelfAttention(self.config[-1])

        # sub-sampling blocks
        self.dwn_block1 = nn.Sequential(nn.Conv2d(in_channels, self.config[0], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[0]),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(self.config[0], self.config[0], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[0]),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d(2)
                                        )

        self.dwn_block2 = nn.Sequential(nn.Conv2d(self.config[0], self.config[1], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[1]),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(self.config[1], self.config[1], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[1]),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d(2)
                                        )

        self.dwn_block3 = nn.Sequential(nn.Conv2d(self.config[1], self.config[2], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[2]),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(self.config[2], self.config[2], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[2]),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d(2)
                                        )

        self.dwn_block4 = nn.Sequential(nn.Conv2d(self.config[2], self.config[3], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[3]),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(self.config[3], self.config[3], kernel, padding=1),
                                        nn.BatchNorm2d(self.config[3]),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d(2)
                                        )

        # up-sampling blocks
        self.up_block1 = nn.Sequential(nn.ConvTranspose2d(self.config[3], self.config[3], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[3]),
                                       nn.LeakyReLU(),
                                       nn.ConvTranspose2d(self.config[3], self.config[3], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[3]),
                                       nn.LeakyReLU())
        self.up_layer1 = nn.ConvTranspose2d(self.config[3], self.config[3], kernel, stride=2, output_padding=1)

        self.up_block2 = nn.Sequential(nn.ConvTranspose2d(3 * self.config[2], self.config[2], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[2]),
                                       nn.LeakyReLU(),
                                       nn.ConvTranspose2d(self.config[2], self.config[2], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[2]),
                                       nn.LeakyReLU())
        self.up_layer2 = nn.ConvTranspose2d(self.config[2], self.config[2], kernel, stride=2, output_padding=1)

        self.up_block3 = nn.Sequential(nn.ConvTranspose2d(3 * self.config[1], self.config[1], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[1]),
                                       nn.LeakyReLU(),
                                       nn.ConvTranspose2d(self.config[1], self.config[1], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[1]),
                                       nn.LeakyReLU())
        self.up_layer3 = nn.ConvTranspose2d(self.config[1], self.config[1], kernel, stride=2, output_padding=1)

        self.up_block4 = nn.Sequential(nn.ConvTranspose2d(3 * self.config[0], self.config[0], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[0]),
                                       nn.LeakyReLU(),
                                       nn.ConvTranspose2d(self.config[0], self.config[0], kernel, padding=1),
                                       nn.BatchNorm2d(self.config[0]),
                                       nn.LeakyReLU())
        self.up_layer4 = nn.ConvTranspose2d(self.config[0], self.config[0], kernel, stride=2, output_padding=1)

        self.out = nn.Sequential(nn.Conv2d(self.config[0], classes, kernel, padding=1),
                                 nn.LogSoftmax(dim=1))

    def squeeze_and_out(self, x, in_channels, scores):
        inc_block = nn.Conv2d(in_channels, self.config[-1], 1, device='cuda')
        dec_block = nn.Conv2d(self.config[-1], in_channels, 1, device='cuda')

        return dec_block(inc_block(x) * scores)

    def forward(self, x):
        x_1 = self.dwn_block1(x)  # (16, 16, 128, 128)
        x_2 = self.dwn_block2(x_1)  # (16, 32, 64, 64)
        x_3 = self.dwn_block3(x_2)  # (16, 64, 32, 32)
        x_4 = self.dwn_block4(x_3)  # (16, 128, 16, 16)

        scores = self.self_attention(x_4)
        scr_up_smpl2 = F.interpolate(scores, scale_factor=2)
        scr_up_smpl4 = F.interpolate(scores, scale_factor=4)
        scr_up_smpl8 = F.interpolate(scores, scale_factor=8)

        x_5 = self.up_block1(self.squeeze_and_out(x_4, self.config[3], scores))
        x_6 = self.up_layer1(x_5)

        pad_x6 = ReplicationPad2d((0, x_3.size(3) - x_6.size(3), 0, x_3.size(2) - x_6.size(2)))
        x_6 = torch.cat((pad_x6(x_6), x_3), dim=1)  # (16, 192, 32, 32)

        x_7 = self.up_block2(self.squeeze_and_out(x_6, 3 * self.config[2], scr_up_smpl2))
        x_8 = self.up_layer2(x_7)

        pad_x8 = ReplicationPad2d((0, x_2.size(3) - x_8.size(3), 0, x_2.size(2) - x_8.size(2)))
        x_8 = torch.cat((pad_x8(x_8), x_2), dim=1)  # (16, 96, 64, 64)

        x_9 = self.up_block3(self.squeeze_and_out(x_8, 3 * self.config[1], scr_up_smpl4))
        x_10 = self.up_layer3(x_9)

        pad_x10 = ReplicationPad2d((0, x_1.size(3) - x_10.size(3), 0, x_1.size(2) - x_10.size(2)))
        x_11 = torch.cat((pad_x10(x_10), x_1), dim=1)  # (16, 48, 128, 128)

        x_12 = self.up_block4(self.squeeze_and_out(x_11, 3 * self.config[0], scr_up_smpl8))
        x_13 = self.up_layer4(x_12)

        pad_x12 = ReplicationPad2d((0, x.size(3) - x_13.size(3), 0, x.size(2) - x_13.size(2)))
        x_13 = self.out(pad_x12(x_13))  # (16, 16, 258, 258)

        return x_13


if __name__ == "__main__":
    t = torch.randn(16, 6, 256, 256)
    t_ = torch.randn(16, 128, 16, 16)

    model = FCFEWithAttention(6, 2)
    out = model(t)

    print(out.shape)

    # att = SelfAttention(128)
    # print(att(t_).shape)
