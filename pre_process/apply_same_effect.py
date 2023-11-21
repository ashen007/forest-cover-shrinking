import random
import numpy as np
import torch

from torchvision.transforms.v2 import GaussianBlur
from torchvision.transforms.functional import *


class ColorJitter(object):

    def __init__(self, p: float, factor: tuple):
        self.p = p
        self.factor = random.uniform(factor[0], factor[1])
        self.order = ['bri', 'cont'] # 'sat', 'hue'

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            changes = random.sample(self.order, np.random.randint(1, len(self.order)))

            for c in changes:
                if c == 'bri':
                    img1 = adjust_brightness(img1, self.factor)
                    img2 = adjust_brightness(img2, self.factor)

                elif c == 'sat':
                    img1 = adjust_saturation(img1, self.factor)
                    img2 = adjust_saturation(img2, self.factor)

                elif c == 'hue':
                    img1 = adjust_saturation(img1, self.factor)
                    img2 = adjust_saturation(img2, self.factor)

                else:
                    img1 = adjust_contrast(img1, self.factor)
                    img2 = adjust_contrast(img2, self.factor)

        return img1, img2, label


class GaussianBlur2(object):

    def __init__(self, kernel, p):
        self.p = p
        self.blur = GaussianBlur(kernel)

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            img1 = self.blur(img1)
            img2 = self.blur(img2)

        return img1, img2, label


class RandomInvert(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            img1 = invert(img1)
            img2 = invert(img2)

        return img1, img2, label


class RandomEqualize(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            img1 = equalize(img1.to(torch.uint8)).to(torch.float32)
            img2 = equalize(img2.to(torch.uint8)).to(torch.float32)

        return img1, img2, label


class RandomAdjustSharpness(object):

    def __init__(self, factor, p):
        self.factor = factor
        self.p = p

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            img1 = adjust_sharpness(img1, self.factor)
            img2 = adjust_sharpness(img2, self.factor)

        return img1, img2, label


class RandomRotation(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, sample, *args, **kwargs):
        img1, img2, label = sample

        if np.random.random() > self.p:
            by = np.random.randint(0, 3)
            img1 = torch.rot90(img1, by, dims=(1, 2))
            img2 = torch.rot90(img2, by, dims=(1, 2))
            label = torch.rot90(label, by, dims=(0, 1))

        return img1.squeeze(0), img2.squeeze(0), label.squeeze(0)
