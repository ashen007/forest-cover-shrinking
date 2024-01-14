import os
import pandas as pd
import torch
import numpy as np
import tifffile

from math import ceil
from tqdm import tqdm
from skimage.io import imread

ROOT = '../data/OSCD/Onera Satellite Change Detection dataset/'
SIDE = 96
STRIDE = (SIDE // 2) - 1
FP_MODIFIER = 10


def adjust_shape(I, s):
    """Adjust the shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = imread(path + im_name + "B04.tif")
    g = imread(path + im_name + "B03.tif")
    b = imread(path + im_name + "B02.tif")

    I = np.stack((r, g, b), axis=2).astype('float')
    I = (I - I.mean()) / I.std()

    return I


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


def read_sentinel_img_trio(path):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    I1 = read_sentinel_img(path + '/imgs_1/')
    I2 = read_sentinel_img(path + '/imgs_2/')

    cm = imread(path + '/cm/cm.png', as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

    return I1, I2, cm


def get_names(path):
    return pd.read_csv(path + 'train.txt').columns


def oscd_dataset(names, path=ROOT, patch_side=SIDE, stride=STRIDE):
    imgs_1 = {}
    imgs_2 = {}
    cms = {}
    n_patch_per_img = {}
    n_patches = 0
    patch_coords = []
    n_pix = 0
    true_pix = 0

    for name in tqdm(names):
        I1, I2, cm = read_sentinel_img_trio(path + name)
        imgs_1[name] = reshape_for_torch(I1)
        imgs_2[name] = reshape_for_torch(I2)
        cms[name] = cm

        # for weight calculation
        s = cm.shape
        n_pix += np.prod(s)
        true_pix += cm.sum()

        # calculate the number of patches
        s = imgs_1[name].shape
        n1 = ceil((s[1] - patch_side + 1) / stride)
        n2 = ceil((s[2] - patch_side + 1) / stride)
        n_patches_i = n1 * n2
        n_patch_per_img[name] = n_patches_i
        n_patches += n_patches_i

        for i in range(n1):
            for j in range(n2):
                # coordinates in (x1, x2, y1, y2)
                current_patch_coords = (name,
                                        [stride * i, stride * i + patch_side, stride * j,
                                         stride * j + patch_side],
                                        [stride * (i + 1), stride * (j + 1)])
                patch_coords.append(current_patch_coords)

    weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    return imgs_1, imgs_2, cms, patch_coords, weights, names


def create_patched(dst):
    img1s, img2s, cms, patch_coors, w, _ = oscd_dataset(get_names(ROOT))

    # save weights
    w = torch.from_numpy(np.asarray(w)).float()
    torch.save(w, "../data/OSCD/class_weight.pt")

    for i, coord in tqdm(enumerate(patch_coors)):
        name, limit, c = coord

        I1 = img1s[name][:, limit[0]:limit[1], limit[2]:limit[3]]
        I2 = img2s[name][:, limit[0]:limit[1], limit[2]:limit[3]]

        label = cms[name][limit[0]:limit[1], limit[2]:limit[3]]

        I1 = torch.from_numpy(np.asarray(I1)).to(torch.float32)
        I2 = torch.from_numpy(np.asarray(I2)).to(torch.float32)
        label = torch.from_numpy(1 * np.array(label)).float()

        img = torch.cat((I1, I2, label.unsqueeze(0)), dim=0)

        dst_path = os.path.join(dst, f'{name}_{i}.tiff')
        tifffile.imwrite(dst_path, img.numpy())


if __name__ == '__main__':
    create_patched('../data/OSCD/annotated')
