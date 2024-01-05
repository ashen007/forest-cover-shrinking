import torch

from torch import nn
from torch.nn import functional as F
from forest_cover_change_detection.models.fcef.modules import ResNeStBlock


class StageZero(nn.Module):
    """for level: 128, 64 dimensions"""

    def __init__(self, in_channels, out_channels, p=1, r=1, t=2):
        super(StageZero, self).__init__()

        pre_process_layers = []
        trunk_layers = []
        inter_channels = in_channels

        self.mx_pool = nn.MaxPool2d(2)

        # pre-process layers
        for _ in range(p):
            pre_process_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # trunk layers
        inter_channels = in_channels

        for _ in range(t):
            trunk_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # soft mask branch
        mask_head_layers = [nn.MaxPool2d(2)]
        mask_tail_layers = []

        for _ in range(r):
            mask_head_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))
            mask_tail_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))

        tail_layers = [nn.Upsample(scale_factor=2),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Sigmoid()
                       ]
        mask_tail_layers += tail_layers

        # down
        down_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_1 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_1_layers)

        down_2_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_2 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_2_layers)

        down_3_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_3 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_3_layers)

        down_4_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_4 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_4_layers)

        # up
        up_4_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_4 = nn.Sequential(*up_4_layers,
                                  nn.Upsample(scale_factor=2))

        up_3_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_3 = nn.Sequential(*up_3_layers,
                                  nn.Upsample(scale_factor=2))

        up_2_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_2 = nn.Sequential(*up_2_layers,
                                  nn.Upsample(scale_factor=2))

        up_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_1 = nn.Sequential(*up_1_layers,
                                  nn.Upsample(scale_factor=2))

        # assembling
        self.pre_process = nn.Sequential(*pre_process_layers)
        self.trunk = nn.Sequential(*trunk_layers)
        self.mask_head = nn.Sequential(*mask_head_layers)
        self.mask_tail = nn.Sequential(*mask_tail_layers)
        self.post_process = nn.Sequential(*pre_process_layers)

    def forward(self, x):
        x = self.pre_process(x)
        trunk_branch = self.trunk(x)

        # mask branch
        x_ = self.mask_head(x)
        d1 = self.down_1(x_)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)

        u4 = self.up_4(d4)

        u4 += d3
        u3 = self.up_3(u4)

        u3 += d2
        u2 = self.up_2(u3)

        u2 += d1
        u1 = self.up_1(u2)

        u1 += x_
        out_ = self.mask_tail(u1)

        # assembling
        out_ = (trunk_branch * out_) + trunk_branch

        return self.post_process(out_)


class StageOne(nn.Module):
    """for level: 32 dimensions"""

    def __init__(self, in_channels, out_channels, p=1, r=1, t=2):
        super(StageOne, self).__init__()

        pre_process_layers = []
        trunk_layers = []
        inter_channels = in_channels

        self.mx_pool = nn.MaxPool2d(2)

        # pre-process layers
        for _ in range(p):
            pre_process_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # trunk layers
        inter_channels = in_channels

        for _ in range(t):
            trunk_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # soft mask branch
        mask_head_layers = [nn.MaxPool2d(2)]
        mask_tail_layers = []

        for _ in range(r):
            mask_head_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))
            mask_tail_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))

        tail_layers = [nn.Upsample(scale_factor=2),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Sigmoid()
                       ]
        mask_tail_layers += tail_layers

        # down
        down_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_1 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_1_layers)

        down_2_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_2 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_2_layers)

        # up
        up_2_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_2 = nn.Sequential(*up_2_layers,
                                  nn.Upsample(scale_factor=2))

        up_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_1 = nn.Sequential(*up_1_layers,
                                  nn.Upsample(scale_factor=2))

        # assembling
        self.pre_process = nn.Sequential(*pre_process_layers)
        self.trunk = nn.Sequential(*trunk_layers)
        self.mask_head = nn.Sequential(*mask_head_layers)
        self.mask_tail = nn.Sequential(*mask_tail_layers)
        self.post_process = nn.Sequential(*pre_process_layers)

    def forward(self, x):
        x = self.pre_process(x)
        trunk_branch = self.trunk(x)

        # mask branch
        x_ = self.mask_head(x)
        d1 = self.down_1(x_)
        d2 = self.down_2(d1)

        u2 = self.up_2(d2)

        u2 += d1
        u1 = self.up_1(u2)

        u1 += x_
        out_ = self.mask_tail(u1)

        # assembling
        out_ = (trunk_branch * out_) + trunk_branch

        return self.post_process(out_)


class StageTwo(nn.Module):
    """for level: 32 dimensions"""

    def __init__(self, in_channels, out_channels, p=1, r=1, t=2):
        super(StageTwo, self).__init__()

        pre_process_layers = []
        trunk_layers = []
        inter_channels = in_channels

        self.mx_pool = nn.MaxPool2d(2)

        # pre-process layers
        for _ in range(p):
            pre_process_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # trunk layers
        inter_channels = in_channels

        for _ in range(t):
            trunk_layers.append(ResNeStBlock(inter_channels, out_channels, 2, 2, 40))
            inter_channels = out_channels

        # soft mask branch
        mask_head_layers = [nn.MaxPool2d(2)]
        mask_tail_layers = []

        for _ in range(r):
            mask_head_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))
            mask_tail_layers.append(ResNeStBlock(inter_channels, inter_channels, 2, 2, 40))

        tail_layers = [nn.Upsample(scale_factor=2),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Conv2d(inter_channels, inter_channels, 1),
                       nn.Sigmoid()
                       ]
        mask_tail_layers += tail_layers

        # down
        down_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.down_1 = nn.Sequential(nn.MaxPool2d(2),
                                    *down_1_layers)

        up_1_layers = [ResNeStBlock(inter_channels, inter_channels, 2, 2, 40) for _ in range(r)]
        self.up_1 = nn.Sequential(*up_1_layers,
                                  nn.Upsample(scale_factor=2))

        # assembling
        self.pre_process = nn.Sequential(*pre_process_layers)
        self.trunk = nn.Sequential(*trunk_layers)
        self.mask_head = nn.Sequential(*mask_head_layers)
        self.mask_tail = nn.Sequential(*mask_tail_layers)
        self.post_process = nn.Sequential(*pre_process_layers)

    def forward(self, x):
        x = self.pre_process(x)
        trunk_branch = self.trunk(x)

        # mask branch
        x_ = self.mask_head(x)
        d1 = self.down_1(x_)

        print(d1.shape)

        u1 = self.up_1(d1)

        u1 += x_
        out_ = self.mask_tail(u1)

        # assembling
        out_ = (trunk_branch * out_) + trunk_branch

        return self.post_process(out_)


if __name__ == '__main__':
    t = torch.randn(4, 32, 16, 16)
    m = StageTwo(32, 32, 2)

    print(m(t).shape)
