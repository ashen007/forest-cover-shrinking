import random

import torch
import numpy as np


def random_flip(images):
    img1, img2, label = images

    if np.random.random() > 0.5:
        img1 = torch.from_numpy(img1.numpy()[:, :, ::-1].copy())
        img2 = torch.from_numpy(img2.numpy()[:, :, ::-1].copy())
        label = torch.from_numpy(label.numpy()[:, ::-1].copy())

    return img1, img2, label


def random_rotate(images):
    img1, img2, label = images
    n = np.random.randint(0, 3)

    if np.random.random() > 0.5:
        img1 = torch.from_numpy(np.rot90(img1.numpy(), n, axes=(1, 2)).copy())
        img2 = torch.from_numpy(np.rot90(img2.numpy(), n, axes=(1, 2)).copy())
        label = torch.from_numpy(np.rot90(label.numpy(), n, axes=(1, 2)).copy())

    return img1, img2, label
