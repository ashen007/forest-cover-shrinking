import warnings

warnings.filterwarnings(action='ignore')

import os.path
import pandas as pd
import torch

from tqdm import tqdm
from torch.nn import ModuleList
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import v2
from pre_process.augmentations import GenerateRandomAugmentationDataset


class RandomCropWithMask(GenerateRandomAugmentationDataset):
    TRANSFORMS = v2.Compose([v2.RandomCrop(300),
                             v2.Resize(480)])

    def __init__(self, annotation_file):
        super().__init__(annotation_file)

    @staticmethod
    def merge(img1, img2, label):
        return torch.cat((img1, img2, label), dim=0)

    def apply_random_transformation(self, image, file_names, copies=3):
        trans_copies = {}

        for i in range(copies):
            img = self.TRANSFORMS(image)

            trans_copies[(f"c_{i}_{file_names[0]}",
                          f"c_{i}_{file_names[1]}",
                          f"c_{i}_{file_names[2]}")] = [img[:3, ::], img[3:6, ::], img[6:, ::]]

        return trans_copies

    def save_images(self, names, images, directory, annotation_directory):
        for file_names, images in zip(names, images):
            write_jpeg(images[0], os.path.join(self.ROOT, directory, file_names[0]))
            write_jpeg(images[1], os.path.join(self.ROOT, directory, file_names[1]))
            write_jpeg(images[2], os.path.join(self.ROOT, directory, annotation_directory, file_names[2]))

    def do_transformation(self):
        for d in tqdm(self.DIRS, desc='directories'):
            self.update_file_list(d)

            for _, row in self.file_list[['img_1', 'img_2', 'label']].iterrows():
                names = [row['img_1'], row['img_2'], row['label'].split('\\')[-1]]

                img = self.merge(read_image(os.path.join(self.ROOT, d, row['img_1'])),
                                 read_image(os.path.join(self.ROOT, d, row['img_2'])),
                                 read_image(os.path.join(self.ROOT, row['label']))
                                 )
                copies = self.apply_random_transformation(img, names)

                self.annotated_dataset = pd.concat((self.annotated_dataset,
                                                    pd.DataFrame({'img_1': [i[0] for i in copies.keys()],
                                                                  'img_2': [i[1] for i in copies.keys()],
                                                                  'label': [
                                                                      os.path.join(*row['label'].split('\\')[:-1],
                                                                                   i[2]) for i in copies.keys()],
                                                                  'dir': [d] * len(copies)
                                                                  },
                                                                 columns=self.file_list.columns
                                                                 )),
                                                   ignore_index=True)

                self.save_images(copies.keys(), copies.values(), d, row['label'].split('\\')[-2])

        self.annotated_dataset.to_csv('../data/train_temp.csv', index=False)


if __name__ == "__main__":
    generator = RandomCropWithMask('../data/new_train.csv')
    generator.do_transformation()
