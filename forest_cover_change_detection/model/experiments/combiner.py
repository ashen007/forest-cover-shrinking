import torch
from torch import nn
from ..attention import FeatureExtractor


class Combiner(nn.Module):
    """
    mechanism to apply attention to architecture
    """

    def __init__(self, feature_extractor):
        super(Combiner, self).__init__()

        self.feature_extractor = feature_extractor
        self.weight_selections =

    def forward(self, x):
        features = self.feature_selections(x)

        if len()
