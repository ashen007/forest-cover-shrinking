import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


class WeightSelector(nn.Module):
    """
    dot score to apply attention which is important
    """

    def __init__(self, h):
        super(WeightSelector, self).__init__()

        self.h = h

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)  # shape [16, 9, 64]
        x_context = torch.mean(x, dim=1)  # shape [16, 64]
        scores = torch.bmm(x, x_context.unsqueeze(2)) / np.sqrt(self.h)  # shape [16, 9, 1]
        weights = F.softmax(scores, dim=1)  # shape [16, 9, 1]

        return torch.sum(x * weights, dim=1)  # shape [16, 64]


if __name__ == "__main__":
    t = torch.randn(16, 9, 64, 1, 1)
    ws = WeightSelector(64)

    # shape [16, 9, 1]
    t_ = ws(t)

    print(t_.shape)
