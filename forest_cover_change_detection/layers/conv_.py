import torch
from torch import nn


class IndividualConv2d(nn.Module):
    """
    convolutional layer to apply in (B, T, C, H, W) shape input
    along T dimension
    """

    def __init__(self,
                 in_channels, out_channels, depth,
                 kernel, padding, stride):
        super(IndividualConv2d, self).__init__()

        self.depth = depth
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding)

    def forward(self, x):
        x_is = []

        for i in range(self.depth):
            x_i = self.conv(x[:, i, ::])
            x_is.append(x_i.unsqueeze(1))

        return torch.cat(x_is, dim=1)


class IndividualConv1d(nn.Module):
    """
    convolutional layer to apply in (B, T, C, H, W) shape input
    along T dimension
    """

    def __init__(self, in_channels, out_channels, depth):
        super(IndividualConv1d, self).__init__()

        self.depth = depth
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x_is = []

        for i in range(self.depth):
            x_i = self.conv(x[:, i, ::].unsqueeze(1))
            x_is.append(x_i.unsqueeze(1))

        return torch.cat(x_is, dim=1)


if __name__ == "__main__":
    t = torch.randn(16, 9, 1, 28, 28)
    f = torch.randn(16, 9, 64, 1, 1)
    size = f.size()
    f = f.view(*size[:2], -1) # shape [16, 9, 64]
    conv_t = IndividualConv2d(1, 8, 9, 3, 1, 1)
    conv_f = IndividualConv1d(1, 4, 9)

    # output shape [16, 9, 8, 28, 28]
    t_ = conv_t(t)
    # shape [16, 9, 4, 64]
    f_ = conv_f(f)

    print(t_.shape)
    print(f_.shape)
