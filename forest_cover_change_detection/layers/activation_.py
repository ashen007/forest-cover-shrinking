from torch import nn
from torch.nn import functional as F


class RSoftMax(nn.Module):

    def __init__(self, g=1, r=2):
        super(RSoftMax, self).__init__()

        self.g = g
        self.r = r

    def forward(self, x):
        size = x.shape
        x = x.view(size, self.g, self.r, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(size, -1, 1, 1)

        return x
