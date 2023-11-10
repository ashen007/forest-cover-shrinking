import os
import shutil
import numpy as np

from tqdm import tqdm
from image_cut import get_dir_list, DST_ROOT


def split_train_test_validation(test_size: float = 0.2, val_size: float = 0.2) -> tuple:
    dir_list = get_dir_list(DST_ROOT)
    np.random.shuffle(dir_list)

    train_ = int(len(dir_list) * (1.0 - (test_size + val_size)))
    test_ = int(len(dir_list) * test_size)

    return dir_list[:train_], dir_list[train_: (train_ + test_)], dir_list[(train_ + test_):]


def do_move(splits) -> None:
    train, test, valid = splits
    subsets = {"train": train, "test": test, "valid": valid}

    for subset in ["train", "test", "valid"]:
        if not os.path.isdir(os.path.join(DST_ROOT, subset)):
            os.mkdir(os.path.join(DST_ROOT, subset))

        for d in tqdm(subsets[subset], desc="moving"):
            src = os.path.join(DST_ROOT, d)
            dst = os.path.join(DST_ROOT, subset, d)

            shutil.move(src, dst)


if __name__ == "__main__":
    do_move(split_train_test_validation())
