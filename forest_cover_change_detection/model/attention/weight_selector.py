import torch
from torch import nn
from forest_cover_change_detection.layers.conv_ import IndividualConv1d


class WeightSelector(nn.Module):
    """
    weight selections for pay attention which are important
    """

    def __init__(self, in_channels, out_channels, h_units, depth):
        super(WeightSelector, self).__init__()

        self.weight_selection = nn.Sequential(IndividualConv1d(in_channels, out_channels, depth),
                                              nn.ReLU()
                                              )
        self.out = nn.Sequential(nn.Linear(h_units * out_channels, 1),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        x = self.weight_selection(x)

        size = x.size()
        x = x.view(*size[:2], -1)
        x = self.out(x)

        return x


if __name__ == "__main__":
    t = torch.randn(16, 9, 64, 1, 1)
    ws = WeightSelector(1, 4, 64, 9)

    # shape [16, 9, 1]
    t_ = ws(t)

    print(t_.shape)