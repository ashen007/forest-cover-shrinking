import os
import numpy as np

from tqdm import tqdm
from PIL import Image

ROOT = "../data/raw_data/"
DST_ROOT = "../data/cleaned/"


def get_dir_list(root: str) -> list:
    return os.listdir(root)


def read_img(path: str, resize_to: tuple) -> np.ndarray:
    return np.asarray(Image.open(path).resize(resize_to))


def get_size(img: np.ndarray) -> tuple:
    return img[:img.shape[0] - 60, ::].shape


def create_restriction(w: int, h: int, padding: int, patch_size: tuple) -> tuple:
    assert isinstance(patch_size, (list, tuple)) & (len(patch_size) == 2)
    assert patch_size[0] == patch_size[1]
    mid_point = patch_size[0] // 2

    return (mid_point + padding,
            (w - (mid_point + padding)),
            mid_point + padding,
            (h - (mid_point + padding)))


def get_rand_centers(restriction: tuple, n_anchors: int) -> list:
    x_start, x_end, y_start, y_end = restriction
    selected_mid_anchor = [(np.random.randint(x_start, x_end),
                            np.random.randint(y_start, y_end)) for _ in range(n_anchors)]
    return selected_mid_anchor


def do_cut(img: np.ndarray, mid_point: tuple, patch_size: tuple) -> np.ndarray:
    x, y = mid_point
    c = patch_size[0] // 2

    if img.ndim == 3:
        return img[:, (y - c):(y + c), (x - c):(x + c)]

    else:
        return img[:, (y - c):(y + c), (x - c):(x + c), :]


if __name__ == "__main__":
    PATCH_SIZE = (480, 480)
    N_ANCHORS = 3

    img_cuts = []
    dir_list = get_dir_list(ROOT)
    w, h = (1920, 936)

    for dir in tqdm(dir_list):
        anchors = get_rand_centers(create_restriction(w, h, 100, PATCH_SIZE), N_ANCHORS)
        t = np.asarray(
            [read_img(os.path.join(ROOT, dir, file), resize_to=(w, h)) for file in os.listdir(os.path.join(ROOT, dir))])
        cuts = [do_cut(t, s, PATCH_SIZE) for s in anchors]

        for i in range(N_ANCHORS):
            sub_cut_dir = f"{dir}_{i}"
            sub_path = os.path.join(DST_ROOT, sub_cut_dir)

            if not os.path.isdir(sub_path):
                os.mkdir(sub_path)

            for j in range(len(os.listdir(os.path.join(ROOT, dir)))):
                dst_file_path = os.path.join(sub_path, f"{j}.jpg")
                file = Image.fromarray(cuts[i][j])
                file.save(dst_file_path)
