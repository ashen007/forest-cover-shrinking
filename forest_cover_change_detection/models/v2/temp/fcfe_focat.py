import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import UpSample, DownSample
from forest_cover_change_detection.models.fcfe_with_att.modules import FocusAttentionGate


class FCEFVanilaWithFocAt(nn.Module):
    """
    base architecture, testing different tactics to improve
    feature selection and extractions while decoder is fixed
    """

    def __init__(self, in_channels, classes, kernel=3):
        super(FCEFVanilaWithFocAt, self).__init__()
        filters = [16, 32, 64, 128, 256]

        # down sampling
        # this block are the changing sections, can try different feature extractors
        self.feat_ext_block_1 = DownSample(in_channels, filters[0], dropout=False, blocks=2)
        self.dwn_block_1 = nn.MaxPool2d(2)  # (16, 128, 128)
        # attention gate will use output of the above down-sampled layer

        # these blocks are the changing sections, can try different feature extractors
        self.feat_ext_block_2 = DownSample(filters[0], filters[1], dropout=False, blocks=2)
        self.dwn_block_2 = nn.MaxPool2d(2)  # (32, 64, 64)
        # attention gate will use output of the above down-sampled layer

        # these blocks are the changing sections, can try different feature extractors
        self.feat_ext_block_3 = DownSample(filters[1], filters[2], dropout=False, blocks=2)
        self.dwn_block_3 = nn.MaxPool2d(2)  # (64, 32, 32)
        # attention gate will use output of the above down-sampled layer

        # these blocks are the changing sections, can try different feature extractors
        self.feat_ext_block_4 = DownSample(filters[2], filters[3], dropout=False, blocks=2)
        self.dwn_block_4 = nn.MaxPool2d(2)  # (128, 16, 16)
        # attention gate will use output of the above down-sampled layer

        # these blocks are the changing sections, can try different feature extractors
        self.feat_ext_block_5 = DownSample(filters[3], filters[4], dropout=False, blocks=2)
        self.dwn_block_5 = nn.MaxPool2d(2)  # (256, 8, 8)
        # attention gate will use output of the above down-sampled layer

        # up sampling
        # this is a common up-sample block for all models
        self.up_feat_ext_block_1 = UpSample(filters[4], filters[4], kernel, padding=1, blocks=2)
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_1 = UpSample(filters[4], filters[4], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (256, 16, 16)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_2 = UpSample(3 * filters[3], filters[4], kernel, padding=1, blocks=2)
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_2 = UpSample(filters[4], filters[3], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (128, 32, 32)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_3 = UpSample(3 * filters[2], filters[3], kernel, padding=1, blocks=2)
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_3 = UpSample(filters[3], filters[2], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (64, 64, 64)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_4 = UpSample(3 * filters[1], filters[2], kernel, padding=1, blocks=2)
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_4 = UpSample(filters[2], filters[1], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (32, 128, 128)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_5 = UpSample(3 * filters[0], filters[1], kernel, padding=1, blocks=2)
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_5 = UpSample(filters[1], filters[0], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (16, 256, 256)

        self.out_block = nn.Sequential(nn.Conv2d(filters[0], classes, kernel, padding=1),
                                       nn.LogSoftmax(dim=1))  # (2, 256, 256)

    def forward(self, x):  # (16, 6, 96, 96)
        s_0 = x.shape

        xd_1_fet_ext = self.feat_ext_block_1(x)
        xd_1 = self.dwn_block_1(xd_1_fet_ext)  # 1(6, 16, 48, 48)
        s_1 = xd_1.shape

        xd_2_fet_ext = self.feat_ext_block_2(xd_1)
        xd_2 = self.dwn_block_2(xd_2_fet_ext)  # 1(6, 32, 24, 24)
        s_2 = xd_2.shape

        xd_3_fet_ext = self.feat_ext_block_3(xd_2)
        xd_3 = self.dwn_block_3(xd_3_fet_ext)  # 1(6, 64, 12, 12)
        s_3 = xd_3.shape

        xd_4_fet_ext = self.feat_ext_block_4(xd_3)
        xd_4 = self.dwn_block_4(xd_4_fet_ext)  # (16, 128, 6, 6)
        s_4 = xd_4.shape

        xd_5_fet_ext = self.feat_ext_block_5(xd_4)
        xd_5 = self.dwn_block_5(xd_5_fet_ext)  # (16, 256, 3, 3)
        s_5 = xd_5.shape

        xu_1_b = self.up_feat_ext_block_1(xd_5)
        xu_1 = self.up_block_1(xu_1_b)
        pad_1 = ReplicationPad2d((0, (s_4[3] - xu_1.shape[3]), 0, (s_4[2] - xu_1.shape[2])))
        at_gate_1 = FocusAttentionGate(256, xd_4.shape[1], 2, 1, 1)

        # concat skip connection and up-sampled layer from below
        xd_4_attended = at_gate_1(xd_4, xd_5)
        xu_1 = torch.cat((xd_4_attended, pad_1(xu_1)), dim=1)

        xu_2_b = self.up_feat_ext_block_2(xu_1)
        xu_2 = self.up_block_2(xu_2_b)
        pad_2 = ReplicationPad2d((0, (s_3[3] - xu_2.shape[3]), 0, (s_3[2] - xu_2.shape[2])))
        at_gate_2 = FocusAttentionGate(256, xd_3.shape[1], 4, 1, 3)

        # concat skip connection and up-sampled layer from below
        xd_3_attended = at_gate_2(xd_3, xd_5)
        xu_2 = torch.cat((xd_3_attended, pad_2(xu_2)), dim=1)

        xu_3_b = self.up_feat_ext_block_3(xu_2)
        xu_3 = self.up_block_3(xu_3_b)
        pad_3 = ReplicationPad2d((0, (s_2[3] - xu_3.shape[3]), 0, (s_2[2] - xu_3.shape[2])))
        at_gate_3 = FocusAttentionGate(256, xd_2.shape[1], 8, 1, 7)

        # concat skip connection and up-sampled layer from below
        xd_2_attended = at_gate_3(xd_2, xd_5)
        xu_3 = torch.cat((xd_2_attended, pad_3(xu_3)), dim=1)

        xu_4_b = self.up_feat_ext_block_4(xu_3)
        xu_4 = self.up_block_4(xu_4_b)
        pad_4 = ReplicationPad2d((0, (s_1[3] - xu_4.shape[3]), 0, (s_1[2] - xu_4.shape[2])))
        at_gate_4 = FocusAttentionGate(256, xd_1.shape[1], 16, 1, 15)

        # concat skip connection and up-sampled layer from below
        xd_1_attended = at_gate_4(xd_1, xd_5)
        xu_4 = torch.cat((xd_1_attended, pad_4(xu_4)), dim=1)

        xu_5_b = self.up_feat_ext_block_5(xu_4)
        xu_5 = self.up_block_5(xu_5_b)
        pad_5 = ReplicationPad2d((0, (s_0[3] - xu_5.shape[3]), 0, (s_0[2] - xu_5.shape[2])))

        x_out = self.out_block(pad_5(xu_5))

        return x_out


if __name__ == '__main__':
    t = torch.randn(16, 6, 128, 128).cuda()
    model = FCEFVanilaWithFocAt(6, 2)
    model.cuda()

    print(model(t).shape)
