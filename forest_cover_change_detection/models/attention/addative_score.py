import torch
import torch.nn.functional as F

from torch import nn


class WeightSelector(nn.Module):
    """
    improved dot score with bi-linearity
    """

    def __init__(self, h):
        super(WeightSelector, self).__init__()

        self.w = nn.Linear(2 * h, h)
        self.v = nn.Linear(h, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)  # shape [16, 9, 64]
        x_context = torch.mean(x, dim=1)  # shape [16, 64]
        x_context = torch.stack([x_context for _ in range(size[1])], dim=1)  # shape [16, 9, 64]
        x_cont_combine = torch.cat((x, x_context), dim=2)
        scores = self.v(self.relu(self.w(x_cont_combine)))
        weights = F.softmax(scores, dim=1)  # shape [16, 9, 1]

        return torch.sum(x * weights, dim=1)  # shape [16, 64]


if __name__ == "__main__":
    t = torch.randn(16, 9, 64, 1, 1)
    ws = WeightSelector(64)

    # shape [16, 9, 1]
    t_ = ws(t)

    print(t_.shape)
