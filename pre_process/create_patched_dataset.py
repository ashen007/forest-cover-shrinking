import math
import os.path
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torchvision.io import read_image


def create_patches():
    IMG_SIZE = (3, 480, 480)
    PATCH_SIDE = 96
    STRIDE = (PATCH_SIDE // 2) - 1
    N_PIX = 0
    TRUE_PIX = 0
    N_PATCHES = 0
    ROOT = "../data/annotated/"
    NAMES = pd.read_csv('../data/train.csv')
    NEW_NAMES = pd.DataFrame()
    COLUMNS = ['img_1', 'img_2', 'label', 'dir', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy']

    for idx, row in tqdm(NAMES.iterrows()):
        patch_coords = []
        row_values = [row.img_1, row.img_2, row.label, row.dir]
        label = read_image(os.path.join(ROOT, row.label)).squeeze(0).numpy() != 0

        s = label.shape
        N_PIX += np.prod(s)
        TRUE_PIX += label.sum()

        # calculate the number of patches
        n1 = math.ceil((IMG_SIZE[1] - PATCH_SIDE + 1) / STRIDE)
        n2 = math.ceil((IMG_SIZE[2] - PATCH_SIDE + 1) / STRIDE)
        n_patches_i = n1 * n2
        N_PATCHES += n_patches_i

        # create path co-ordinates
        for i in range(n1):
            for j in range(n2):
                current_patch_coords = ([STRIDE * i, STRIDE * i + PATCH_SIDE, STRIDE * j, STRIDE * j + PATCH_SIDE],
                                        [STRIDE * (i + 1), STRIDE * (j + 1)])
                patch_coords.append(current_patch_coords)

        # create new names dataframe
        new_names = pd.DataFrame([row_values + [*coord, *cen] for coord, cen in patch_coords],
                                 columns=COLUMNS)

        NEW_NAMES = pd.concat((NEW_NAMES, new_names), axis=0)

    weights = [1 * 2 * TRUE_PIX / N_PIX, 2 * (N_PIX - TRUE_PIX) / N_PIX]
    print(TRUE_PIX, N_PIX, weights)

    NEW_NAMES.to_csv("../data/patch_train.csv", index=False)
    torch.save(torch.FloatTensor(weights).cuda(), "../data/class_weight.pt")


if __name__ == "__main__":
    create_patches()
