import torch

from torch import nn
from math import ceil, floor
from torch.nn import functional as F
from torch.autograd import Variable


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


class DynamicKernel(nn.Module):
    def __init__(self, k_size, batch_size, img_size):
        super(DynamicKernel, self).__init__()
        self.k_size = k_size
        self.k_radius = int(self.k_size / 2)
        self.pad = self.k_radius
        self.img_size = img_size + self.k_size - 1
        self.batch = batch_size
        self.kernel = []
        self.dtype = torch.cuda.FloatTensor
        self.dtype_int = torch.cuda.LongTensor

        self.kernel = nn.Conv2d(1, self.k_size * self.k_size, self.k_size, 1,
                                (self.k_radius, self.k_radius)).type(self.dtype)

        self.h = torch.ones((self.k_size, self.k_size)).type(self.dtype_int)
        self.QConv = nn.Conv2d(1, 1, self.k_size, stride=self.k_size).type(self.dtype)

    def forward(self, x):
        x_tmp = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        output = Variable(torch.zeros(x.shape[0], 1, self.img_size, self.img_size).type(self.dtype))
        firstConv = self.kernel(x_tmp)
        for i in range(self.k_size):
            for j in range(self.k_size):
                output += F.pad(x, (self.k_size - 1 - i, i, self.k_size - 1 - j, j)) * torch.tanh(
                    firstConv[:, self.k_size * i + j, :, :].view(x.shape[0], 1, self.img_size, self.img_size))

        return output[:, :, self.pad:-self.pad, self.pad:-self.pad]


class AdaConv2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, batch_size, img_size):
        super(AdaConv2d, self).__init__()
        self.n_kernels_input = n_inputs
        self.n_kernels_output = n_outputs
        self.layer = []
        self.dtype = torch.cuda.FloatTensor

        for j in range(self.n_kernels_output):
            vartmp = []
            for i in range(self.n_kernels_input):
                vartmp.append(DynamicKernel(kernel_size, batch_size, img_size))
            self.layer.append(vartmp)

    def forward(self, x):
        self.output = Variable(torch.zeros(x.shape[0],
                                           self.n_kernels_output,
                                           x.shape[2],
                                           x.shape[3]).type(self.dtype))

        for i in range(self.n_kernels_output):
            for j in range(self.n_kernels_input):
                img_tmp = self.layer[i][j](x[:, j, :, :].view(x.shape[0], 1, x.shape[2], x.shape[3]))
                self.output[:, i, :, :] += img_tmp.view(x.shape[0], x.shape[2], x.shape[3])

        return self.output


if __name__ == "__main__":
    t = torch.randn(16, 9, 1, 28, 28)
    f = torch.randn(16, 9, 64, 1, 1)
    size = f.size()
    f = f.view(*size[:2], -1)  # shape [16, 9, 64]
    conv_t = IndividualConv2d(1, 8, 9, 3, 1, 1)
    conv_f = IndividualConv1d(1, 4, 9)

    # output shape [16, 9, 8, 28, 28]
    t_ = conv_t(t)
    # shape [16, 9, 4, 64]
    f_ = conv_f(f)

    print(t_.shape)
    print(f_.shape)
