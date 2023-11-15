# import csv
import copy
import os
import pandas as pd

ROOT = "../../data/annotated"


def create_pair_list(x: list) -> list:
    pair_list = []
    x_cpy = copy.copy(x)[1:]

    for p in zip(x[:-1], x_cpy):
        pair_list.append(p)

    return pair_list


def sort_files(x: list) -> list:
    sort_file_names = sorted([int(i.split('.')[0]) for i in x])

    return [f"{i}.jpg" for i in sort_file_names]


def create_dataset(x: list, y: list) -> list:
    labeled_pairs = []

    for i in range(len(x)):
        labeled_pairs.append((x[i], y[i]))

    return labeled_pairs


def create_image_pairs(file_path: str) -> None:
    data_labels = []
    sub_dir_list = os.listdir(ROOT)

    for dir in sub_dir_list:
        sub_dir_path = os.path.join(ROOT, dir)
        file_list = os.listdir(sub_dir_path)
        label_folder = file_list.pop(-1)
        pair_list = create_pair_list(sort_files(file_list))

        assert len(pair_list) == len(os.listdir(os.path.join(ROOT, dir, label_folder)))

        data_ = [(int(i.split('-a')[0].split('-')[1]),
                  os.path.join(ROOT, dir, label_folder, i)) for i in os.listdir(os.path.join(ROOT, dir, label_folder))]
        df = pd.DataFrame(data_).sort_values(by=0)

        data_labels += create_dataset(pair_list, df[1].values)

    df = pd.DataFrame(data_labels)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    create_image_pairs('../../data/annotated/train.csv')
