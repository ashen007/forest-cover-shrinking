import os.path
import pandas as pd
import torch

from torchvision.transforms.functional import to_pil_image
from torchvision import io
from torch.utils.data import Dataset, DataLoader
from pre_process.run_time_augmentation import *


class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir, annotation_file, transformers=None, concat=True):
        self.root = root_dir
        self.label_file = pd.read_csv(annotation_file)
        self.transformers = transformers
        self.concat = concat

    def __len__(self) -> int:
        return len(self.label_file)

    def __getitem__(self, idx) -> tuple:
        x_1 = self.label_file.loc[idx, 'img_1']
        x_2 = self.label_file.loc[idx, 'img_2']
        y = self.label_file.loc[idx, 'label']

        img1_path = os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_1)
        img2_path = os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_2)
        label_path = os.path.join(self.root, y)

        x_1_img = io.read_image(img1_path) / 255.0
        x_2_img = io.read_image(img2_path) / 255.0
        y_img = io.read_image(label_path) / 255.0

        x_1_img = x_1_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                  self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]
        x_2_img = x_2_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                  self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]
        y_img = y_img[:, self.label_file.loc[idx, 'x1']:self.label_file.loc[idx, 'x2'],
                self.label_file.loc[idx, 'y1']:self.label_file.loc[idx, 'y2']]

        y_img = y_img.squeeze(0)
        x_1_img_, x_2_img_, y_img_ = random_flip((x_1_img, x_2_img, y_img))

        if self.concat:
            x = torch.cat((x_1_img_, x_2_img_), dim=0)

        return (x, y_img_.long()) if self.concat else (x_1_img_, x_2_img_, y_img_.long())


if __name__ == "__main__":
    data_set = ChangeDetectionDataset('../../data/annotated',
                                      '../../data/train.csv',
                                      )
    data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
    x, y = next(iter(data_loader))

    print(type(x), x.shape)
    print(type(y), y.shape)
