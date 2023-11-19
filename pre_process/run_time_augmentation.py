import torch
import numpy as np

from torchvision.transforms.v2 import RandomCrop

transformer = RandomCrop(256)


def random_flip(images):
    img1, img2, label = images
    axis = {0: [1], 1: [2], 2: [1, 2]}
    n = np.random.randint(0, 3)

    if np.random.random() > 0.5:
        img1 = torch.flip(img1, dims=axis[n])
        img2 = torch.flip(img2, dims=axis[n])
        label = torch.flip(label, dims=[(i - 1) for i in axis[n]])

    return img1, img2, label


def random_rotate(images):
    img1, img2, label = images
    n = np.random.randint(0, 3)

    if n:
        img1 = torch.from_numpy(np.rot90(img1.numpy(), n, axes=(1, 2)).copy())
        img2 = torch.from_numpy(np.rot90(img2.numpy(), n, axes=(1, 2)).copy())
        label = torch.from_numpy(np.rot90(label.numpy(), n, axes=(1, 2)).copy())

    return img1, img2, label


def random_crop(images):
    img1, img2, label = images

    cube = torch.cat((img1, img2, label), dim=0)
    cut = transformer(cube)
    img1, img2, label = cut[:3, ::], cut[3:6, ::], cut[6:, ::]

    return img1, img2, label
