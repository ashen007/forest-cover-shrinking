import torch
from torch import nn
from forest_cover_change_detection.models.attention import FeatureExtractor, Combiner
from forest_cover_change_detection.models.attention.simple_score import WeightSelector as SimpleScore


class AttentionWithoutContex(nn.Module):
    """
    attention model without knowledge of overall contex
    """

    def __init__(self, features: FeatureExtractor, weights: SimpleScore, h_units, classes: int):
        super(AttentionWithoutContex, self).__init__()

        self.features = features
        self.weights = weights
        self.combiner = Combiner(self.features, self.weights)
        self.classifier = nn.Sequential(nn.Linear(self.features.filters[-1], h_units),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(h_units),
                                        nn.Linear(h_units, classes)
                                        )

    def forward(self, x):
        x = self.combiner(x)
        x = self.classifier(x)

        return x


class AttentionWithContex(nn.Module):
    """
    attention model with knowledge of overall contex
    """

    def __init__(self, features: FeatureExtractor, weights, h_units, classes):
        super(AttentionWithContex, self).__init__()

        self.features = features
        self.weights = weights
        self.classifier = nn.Sequential(nn.Linear(self.features.filters[-1], h_units),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(h_units),
                                        nn.Linear(h_units, classes)
                                        )

    def forward(self, x):
        x = self.features(x)
        x = self.weights(x)
        x = self.classifier(x)

        return x
