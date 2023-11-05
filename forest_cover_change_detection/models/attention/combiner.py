import torch
from torch import nn
from forest_cover_change_detection.models.attention import FeatureExtractor
from forest_cover_change_detection.models.attention.simple_score import WeightSelector


class Combiner(nn.Module):
    """
    mechanism to apply attention to architecture
    """

    def __init__(self, feature_extractor, weight_selector):
        super(Combiner, self).__init__()

        self.feature_extractor = feature_extractor
        self.weight_selections = weight_selector

    def forward(self, x):
        features = self.feature_extractor(x)  # shape [16, 9, 64, 1, 1]
        size = features.size()
        features = features.view(*size[:2], -1)  # shape [16, 9, 64]
        weights = self.weight_selections(features)  # shape [16, 9, 1]

        if len(weights.shape) == 2:
            weights.unsqueeze(2)

        scores = features * weights

        return torch.sum(scores, dim=1)


if __name__ == "__main__":
    t = torch.randn(16, 9, 1, 28, 28)
    fe = FeatureExtractor(1, 3, 9)
    ws = WeightSelector(1, 4, 64, 9)
    conn = Combiner(fe, ws)

    t_ = conn(t)

    print(t_.shape)  # shape [16,64]
