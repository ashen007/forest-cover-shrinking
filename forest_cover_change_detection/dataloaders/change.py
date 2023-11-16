import os.path
import pandas as pd
import PIL

from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir, annotation_file, transformers=None):
        self.root = root_dir
        self.label_file = pd.read_csv(annotation_file)
        self.transformers = transformers

    def __len__(self) -> int:
        return len(self.label_file)

    def __getitem__(self, idx) -> tuple:
        x_1 = self.label_file.loc[idx, 'img_1']
        x_2 = self.label_file.loc[idx, 'img_2']
        y = self.label_file.loc[idx, 'label']

        x_1_img = PIL.Image.open(os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_1))
        x_2_img = PIL.Image.open(os.path.join(self.root, self.label_file.loc[idx, 'dir'], x_2))
        y_img = PIL.Image.open(os.path.join(self.root, y))

        if self.transformers is not None:
            x_1_img = self.transformers(x_1_img)
            x_2_img = self.transformers(x_2_img)

        x_1_img = to_tensor(x_1_img) / 255.0
        x_2_img = to_tensor(x_2_img) / 255.0
        y_img = to_tensor(y_img)

        return x_1_img, x_2_img, y_img


if __name__ == "__main__":
    data_set = ChangeDetectionDataset('../../data/annotated', '../../data/train.csv')
    data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
    x_1, x_2, y = next(iter(data_loader))

    print(x_1.shape)
    print(x_2.shape)
    print(y.shape)
