import os.path
import numpy as np
import pandas as pd
import torch

from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader


class DigitClassifierDataset(Dataset):

    def __init__(self, img_dir, labels, volume=3, transforms=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels)
        self.volume = volume
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        select_ids = np.random.randint(0, self.__len__(), size=self.volume)
        select_img_paths = [os.path.join(self.img_dir, self.labels.loc[i, 'file']) for i in select_ids]

        if self.transforms is not None:
            imgs = [io.read_image(path).to(torch.float32) / 255.0 for path in select_img_paths]
            x = torch.stack([self.transforms(transforms.ToPILImage()(img)) for img in imgs])

        else:
            x = torch.stack([io.read_image(path).to(torch.float32) / 255.0 for path in select_img_paths])

        y = max([int(self.labels.loc[i, 'label']) for i in select_ids])

        return x, y


if __name__ == "__main__":
    dataset = DigitClassifierDataset('../../data/exp_data/digit/train',
                                     '../../data/exp_data/digit/train_labels.csv')
    train_loader = DataLoader(dataset, 16, True)

    print(next(iter(train_loader))[0].shape)
