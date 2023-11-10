import os
import shutil

from tqdm import tqdm

# import from custom modules
from pre_process.select_years import select_years


def copy_selected_files_to(years: list, root: str, dst: str) -> None:
    loc_list = os.listdir(root)

    if not os.path.isdir(dst):
        os.mkdir(dst)

    else:
        for d in os.listdir(dst):
            shutil.rmtree(os.path.join(dst, d))

    for i in tqdm(loc_list, desc="coping"):
        src_path = os.path.join(root, i)
        img_list = os.listdir(src_path)
        dst_dir = os.path.join(dst, i)

        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        for img in img_list:
            file_year = int(img[:4])

            if file_year in years:
                src = os.path.join(src_path, img)

                try:
                    shutil.copy(src, os.path.join(dst_dir, img))

                except shutil.Error:
                    pass


if __name__ == "__main__":
    root_ = "../data/raw_data/"
    dst_ = "../data/preprocessed"

    copy_selected_files_to(select_years(35, root_), root_, dst_)
