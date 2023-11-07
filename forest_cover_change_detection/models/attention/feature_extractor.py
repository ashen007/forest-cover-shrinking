import torch
from torch import nn
from forest_cover_change_detection.layers import IndividualConv2d


class FeatureExtractor(nn.Module):
    """
    extract features from input batch of volumetric
    image data
    """

    def __init__(self, in_channels, kernel, depth):
        super(FeatureExtractor, self).__init__()

        self.layers = []
        self.filters = [16, 32, 64]

        for h in self.filters:
            self.layers.append(nn.Sequential(IndividualConv2d(in_channels, h, depth, kernel, padding=1, stride=2),
                                             nn.ReLU()
                                             )
                               )

            in_channels = h

        self.backbone = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.cat([nn.AvgPool2d(3)(x[:, i, ::]).unsqueeze(1) for i in range(x.shape[1])], dim=1)

        return x


if __name__ == "__main__":
    t = torch.randn(16, 9, 1, 28, 28)
    fe = FeatureExtractor(1, 3, 9)

    # output size [16, 9, 64, 1, 1]
    t_ = fe(t)

    print(t_.shape)
