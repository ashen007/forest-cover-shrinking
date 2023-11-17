import warnings

warnings.filterwarnings(action='ignore')

import os.path
import pandas as pd

from tqdm import tqdm
from torch.nn import ModuleList
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import v2, RandomApply, functional as F
from pre_process.image_cut import get_dir_list


class GenerateRandomAugmentationDataset:
    ROOT = "../data/annotated/"
    TRANSFORMS = RandomApply(ModuleList([v2.ColorJitter(),
                                         v2.RandomPhotometricDistort(),
                                         v2.GaussianBlur(15),
                                         v2.RandomInvert(),
                                         v2.RandomPosterize(2),
                                         v2.RandomSolarize(192),
                                         v2.RandomAdjustSharpness(2),
                                         v2.RandomAutocontrast()
                                         ])
                             )
    DIRS = get_dir_list(ROOT)

    def __init__(self, annotation_file):
        self.__file_list = None
        self.annotated_dataset = pd.read_csv(annotation_file)

    def update_file_list(self, dir_path: str) -> None:
        self.__file_list = self.annotated_dataset[self.annotated_dataset['dir'] == dir_path]

    def apply_random_transformation(self, image, file_name, copies=2):
        trans_copies = {}

        for i in range(copies):
            trans_copies[f"a_{i}_{file_name}"] = self.TRANSFORMS(image)

        return list(trans_copies.keys()), trans_copies.values()

    def do_transformation(self):
        for d in tqdm(self.DIRS, desc='directories'):
            self.update_file_list(d)

            for _, row in self.__file_list[['img_1', 'img_2', 'label']].iterrows():
                path1 = os.path.join(self.ROOT, d, row['img_1'])
                path2 = os.path.join(self.ROOT, d, row['img_2'])

                new_names1, img1_aug = self.apply_random_transformation(read_image(path1), row['img_1'])
                new_names2, img2_aug = self.apply_random_transformation(read_image(path2), row['img_2'])

                self.annotated_dataset = pd.concat((self.annotated_dataset,
                                                    pd.DataFrame({'img_1': new_names1,
                                                                  'img_2': new_names2,
                                                                  'label': [row['label']] * len(new_names1),
                                                                  'dir': [d] * len(new_names1)},
                                                                 columns=self.__file_list.columns)),
                                                   ignore_index=True)

                for file_name, img in zip(new_names1, img1_aug):
                    write_jpeg(img,
                               os.path.join(self.ROOT, d, file_name))

                for file_name, img in zip(new_names2, img2_aug):
                    write_jpeg(img,
                               os.path.join(self.ROOT, d, file_name))

        self.annotated_dataset.to_csv('../data/new_train.csv', index=False)


if __name__ == "__main__":
    generator = GenerateRandomAugmentationDataset('../data/train.csv')
    generator.do_transformation()
