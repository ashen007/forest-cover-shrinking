import torch
import torch.nn.functional as F

from torch import nn


class WeightSelector(nn.Module):
    """
    improved dot score with bi-linearity
    """

    def __init__(self, h):
        super(WeightSelector, self).__init__()

        self.weight = nn.Bilinear(h, h, 1)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)  # shape [16, 9, 64]
        x_context = torch.mean(x, dim=1)  # shape [16, 64]
        x_context = torch.stack([x_context for _ in range(size[1])], dim=1)  # shape [16, 9, 64]

        return self.weight(x, x_context)


if __name__ == "__main__":
    t = torch.randn(16, 9, 64, 1, 1)
    ws = WeightSelector(64)

    # shape [16, 9, 1]
    t_ = ws(t)

    print(t_.shape)
