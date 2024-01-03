import torch

from torch import nn
from torch.nn.modules.padding import ReplicationPad2d
from forest_cover_change_detection.models.fcef.modules import ResidualDownSample, UpSample
from forest_cover_change_detection.models.fcef.modules import ResNeStFCA, MultiSpectralAttentionLayer


class FCFEResNeStFCAv2(nn.Module):

    def __init__(self, in_channels, classes, kernel=3):
        super(FCFEResNeStFCAv2, self).__init__()
        filters = [16, 32, 64, 128, 256]
        self.drop = nn.Dropout(0.2)
        self.c2wh = dict([(16, 112), (32, 56), (64, 28), (128, 14), (256, 7)])

        # down sampling
        self.feat_ext_block_1 = nn.Sequential(ResidualDownSample(in_channels, filters[0]),
                                              ResNeStFCA(filters[0], filters[0], 2, 1),
                                              ResNeStFCA(filters[0], filters[0], 2, 1))
        self.dwn_block_1 = nn.MaxPool2d(2)  # (16, 128, 128)

        self.feat_ext_block_2 = nn.Sequential(ResidualDownSample(filters[0], filters[1]),
                                              ResNeStFCA(filters[1], filters[1], 2, 1),
                                              ResNeStFCA(filters[1], filters[1], 2, 1))
        self.dwn_block_2 = nn.MaxPool2d(2)  # (32, 64, 64)

        self.feat_ext_block_3 = nn.Sequential(ResidualDownSample(filters[1], filters[2]),
                                              ResNeStFCA(filters[2], filters[2], 2, 1),
                                              ResNeStFCA(filters[2], filters[2], 2, 1))
        self.dwn_block_3 = nn.MaxPool2d(2)  # (64, 32, 32)

        self.feat_ext_block_4 = nn.Sequential(ResidualDownSample(filters[2], filters[3]),
                                              ResNeStFCA(filters[3], filters[3], 2, 1),
                                              ResNeStFCA(filters[3], filters[3], 2, 1))
        self.dwn_block_4 = nn.MaxPool2d(2)  # (128, 16, 16)

        self.feat_ext_block_5 = nn.Sequential(ResidualDownSample(filters[3], filters[4]),
                                              ResNeStFCA(filters[4], filters[4], 2, 1),
                                              ResNeStFCA(filters[4], filters[4], 2, 1))
        self.dwn_block_5 = nn.MaxPool2d(2)  # (256, 8, 8)

        # up sampling
        self.up_feat_ext_block_1 = nn.Sequential(ResidualDownSample(filters[4], filters[4], kernel),
                                                 ResNeStFCA(filters[4], filters[4], 2, 1),
                                                 ResNeStFCA(filters[4], filters[4], 2, 1))
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_1 = UpSample(filters[4], filters[4], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (256, 16, 16)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_2 = nn.Sequential(ResidualDownSample(3 * filters[3], filters[4], kernel),
                                                 ResNeStFCA(filters[4], filters[4], 2, 1),
                                                 ResNeStFCA(filters[4], filters[4], 2, 1))
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_2 = UpSample(filters[4], filters[3], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (128, 32, 32)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_3 = nn.Sequential(ResidualDownSample(3 * filters[2], filters[3], kernel),
                                                 ResNeStFCA(filters[3], filters[3], 2, 1),
                                                 ResNeStFCA(filters[3], filters[3], 2, 1))
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_3 = UpSample(filters[3], filters[2], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (64, 64, 64)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_4 = nn.Sequential(ResidualDownSample(3 * filters[1], filters[2], kernel),
                                                 ResNeStFCA(filters[2], filters[2], 2, 1),
                                                 ResNeStFCA(filters[2], filters[2], 2, 1))
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_4 = UpSample(filters[2], filters[1], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (32, 128, 128)

        # this is a common up-sample block for all models
        self.up_feat_ext_block_5 = nn.Sequential(ResidualDownSample(3 * filters[0], filters[1], kernel),
                                                 ResNeStFCA(filters[1], filters[1], 2, 1),
                                                 ResNeStFCA(filters[1], filters[1], 2, 1))
        # this block is the layer that increases the dimensions by factor 2
        self.up_block_5 = UpSample(filters[1], filters[0], kernel,
                                   stride=2, padding=0, output_padding=1, blocks=1)  # (16, 256, 256)

        self.out_block = nn.Sequential(nn.Conv2d(filters[0], classes, kernel, padding=1),
                                       nn.LogSoftmax(dim=1))  # (2, 256, 256)

    def forward(self, x_1, x_2):
        s_0 = x_1.shape

        # branch 1
        xd_1_fet_ext_b1 = self.feat_ext_block_1(x_1)
        xd_1_b1 = self.dwn_block_1(xd_1_fet_ext_b1)  # 1(6, 16, 64, 64)
        s_1 = xd_1_b1.shape

        xd_2_fet_ext_b1 = self.feat_ext_block_2(xd_1_b1)
        xd_2_b1 = self.dwn_block_2(xd_2_fet_ext_b1)  # 1(6, 32, 32, 32)
        s_2 = xd_2_b1.shape

        xd_3_fet_ext_b1 = self.feat_ext_block_3(xd_2_b1)
        xd_3_b1 = self.dwn_block_3(xd_3_fet_ext_b1)  # 1(6, 64, 16, 16)
        s_3 = xd_3_b1.shape

        xd_4_fet_ext_b1 = self.feat_ext_block_4(xd_3_b1)
        xd_4_b1 = self.dwn_block_4(xd_4_fet_ext_b1)  # (16, 128, 8, 8)
        s_4 = xd_4_b1.shape

        xd_5_fet_ext_b1 = self.feat_ext_block_5(xd_4_b1)
        xd_5_b1 = self.dwn_block_5(xd_5_fet_ext_b1)  # (16, 256, 4, 4)
        s_5 = xd_5_b1.shape

        # branch 2
        xd_1_fet_ext_b2 = self.feat_ext_block_1(x_2)
        xd_1_b2 = self.dwn_block_1(xd_1_fet_ext_b2)  # 1(6, 16, 64, 64)

        xd_2_fet_ext_b2 = self.feat_ext_block_2(xd_1_b2)
        xd_2_b2 = self.dwn_block_2(xd_2_fet_ext_b2)  # 1(6, 32, 32, 32)

        xd_3_fet_ext_b2 = self.feat_ext_block_3(xd_2_b2)
        xd_3_b2 = self.dwn_block_3(xd_3_fet_ext_b2)  # 1(6, 64, 16, 16)

        xd_4_fet_ext_b2 = self.feat_ext_block_4(xd_3_b2)
        xd_4_b2 = self.dwn_block_4(xd_4_fet_ext_b2)  # (16, 128, 8, 8)

        xd_5_fet_ext_b2 = self.feat_ext_block_5(xd_4_b2)
        xd_5_b2 = self.dwn_block_5(xd_5_fet_ext_b2)  # (16, 256, 4, 4)

        # decoder
        xu_1_b = self.up_feat_ext_block_1((xd_5_b1 - xd_5_b2))
        xu_1 = self.up_block_1(xu_1_b)
        pad_1 = ReplicationPad2d((0, (s_4[3] - xu_1.shape[3]), 0, (s_4[2] - xu_1.shape[2])))
        cat_1 = MultiSpectralAttentionLayer(s_4[1], self.c2wh[s_4[1]], self.c2wh[s_4[1]])

        # concat skip connection and up-sampled layer from below
        xu_1 = torch.cat(((cat_1(xd_4_b1) - cat_1(xd_4_b2)), pad_1(xu_1)), dim=1)

        xu_2_b = self.up_feat_ext_block_2(xu_1)
        xu_2 = self.up_block_2(xu_2_b)
        pad_2 = ReplicationPad2d((0, (s_3[3] - xu_2.shape[3]), 0, (s_3[2] - xu_2.shape[2])))
        cat_2 = MultiSpectralAttentionLayer(s_3[1], self.c2wh[s_3[1]], self.c2wh[s_3[1]])

        # concat skip connection and up-sampled layer from below
        xu_2 = torch.cat(((cat_2(xd_3_b1) - cat_2(xd_3_b2)), pad_2(xu_2)), dim=1)

        xu_3_b = self.up_feat_ext_block_3(xu_2)
        xu_3 = self.up_block_3(xu_3_b)
        pad_3 = ReplicationPad2d((0, (s_2[3] - xu_3.shape[3]), 0, (s_2[2] - xu_3.shape[2])))
        cat_3 = MultiSpectralAttentionLayer(s_2[1], self.c2wh[s_2[1]], self.c2wh[s_2[1]])

        # concat skip connection and up-sampled layer from below
        xu_3 = torch.cat(((cat_3(xd_2_b1) - cat_3(xd_2_b2)), pad_3(xu_3)), dim=1)

        xu_4_b = self.up_feat_ext_block_4(xu_3)
        xu_4 = self.up_block_4(xu_4_b)
        pad_4 = ReplicationPad2d((0, (s_1[3] - xu_4.shape[3]), 0, (s_1[2] - xu_4.shape[2])))
        cat_4 = MultiSpectralAttentionLayer(s_1[1], self.c2wh[s_1[1]], self.c2wh[s_1[1]])

        # concat skip connection and up-sampled layer from below
        xu_4 = torch.cat(((cat_4(xd_1_b1) - cat_4(xd_1_b2)), pad_4(xu_4)), dim=1)

        xu_5_b = self.up_feat_ext_block_5(xu_4)
        xu_5 = self.up_block_5(xu_5_b)
        pad_5 = ReplicationPad2d((0, (s_0[3] - xu_5.shape[3]), 0, (s_0[2] - xu_5.shape[2])))

        x_out = self.out_block(pad_5(xu_5))

        return x_out


if __name__ == '__main__':
    t_1 = torch.randn(16, 3, 128, 128).cuda()
    t_2 = torch.randn(16, 3, 128, 128).cuda()
    model = FCFEResNeStFCAv2(3, 2)
    model.cuda()

    print(model(t_1, t_2).shape)
