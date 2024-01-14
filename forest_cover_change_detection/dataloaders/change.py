import warnings

warnings.filterwarnings(action='ignore')

import os.path
import pandas as pd
import torch

from skimage.io import imread
from torchvision.transforms.v2 import Compose
from torchvision import io
from torch.utils.data import Dataset, DataLoader
from pre_process.run_time_augmentation import *
from pre_process.apply_same_effect import *


class ChangeDetectionDataset(Dataset):
    TRANSFORMS = Compose([
        ColorJitter(0.5, (1.0, 2.0)),
        GaussianBlur(19, 0.5),
        # RandomInvert(0.5),
        # RandomEqualize(0.5),
        RandomAdjustSharpness(2, 0.5),
        RandomRotation(0.5)
    ])

    def __init__(self, root_dir, annotation_file, transformation=True, concat=True, patched=True):
        self.root = root_dir
        self.label_file = pd.read_csv(annotation_file)
        self.transformation = transformation
        self.concat = concat
        self.patched = patched

    def __len__(self) -> int:
        return len(self.label_file)

    def __getitem__(self, idx) -> tuple:
        x_1 = self.label_file.loc[idx, 'img_1']
        x_2 = self.label_file.loc[idx, 'img_2']
        y = self.label_file.loc[idx, 'label']

        img1_path = os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_1)
        img2_path = os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_2)
        label_path = os.path.join(self.root, y)

        x_1_img = io.read_image(img1_path)
        x_2_img = io.read_image(img2_path)
        y_img = io.read_image(label_path) != 0

        x_1_img = (x_1_img - x_1_img.mean()) / x_1_img.std()
        x_2_img = (x_2_img - x_2_img.mean()) / x_2_img.std()

        if self.patched:
            x_1_img = x_1_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                      self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]
            x_2_img = x_2_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                      self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]
            y_img = y_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                    self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]

        else:
            x_1_img, x_2_img, y_img = random_crop((x_1_img, x_2_img, y_img))

        y_img = y_img.squeeze(0)
        x_1_img_, x_2_img_, y_img_ = random_flip((x_1_img, x_2_img, y_img))

        if self.transformation:
            x_1_img_, x_2_img_, y_img_ = self.TRANSFORMS((x_1_img_, x_2_img_, y_img_))

        if self.concat:
            x = torch.cat((x_1_img_, x_2_img_), dim=0)

        return (x, y_img_.long()) if self.concat else ((x_1_img_, x_2_img_), y_img_.long())


class OSCDDataset(Dataset):
    TRANSFORMS = Compose([
        ColorJitter(0.5, (1.0, 2.0)),
        GaussianBlur(19, 0.5),
        # RandomInvert(0.5),
        # RandomEqualize(0.5),
        RandomAdjustSharpness(2, 0.5),
        RandomRotation(0.5)
    ])

    def __init__(self, path, transformation=True):
        self.path = path
        self.img_dir = os.listdir(path)
        self.transformation = transformation

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = imread(os.path.join(self.path, self.img_dir[idx]))
        x1, x2, y = image[:3, ::], image[3:6, ::], image[6:, ::]

        # x1 = x1 / 255.0
        # x2 = x2 / 255.0
        # y = y != 0

        x_1_img, x_2_img, y_img = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y) # random_crop(())
        y_img = y_img.squeeze(0)

        x_1_img_, x_2_img_, y_img_ = random_flip((x_1_img, x_2_img, y_img))

        if self.transformation:
            x_1_img_, x_2_img_, y_img_ = self.TRANSFORMS((x_1_img_, x_2_img_, y_img_))

        return (x_1_img_.to(torch.float), x_2_img_.to(torch.float)), y_img_.long()


if __name__ == "__main__":
    data_set = OSCDDataset('../../data/OSCD/annotated')
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    # data_set = ChangeDetectionDataset('../../data/annotated',
    #                                   '../../data/annotated/train.csv',
    #                                   patched=False,
    #                                   concat=False
    #                                   )
    # data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
    x, y = next(iter(data_loader))

    print(type(x[0]), x[0].shape, x[0].dtype)
    print(type(x[1]), x[1].shape, x[1].dtype)
    print(type(y), y.shape, y.dtype)
