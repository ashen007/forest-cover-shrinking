import os
import torch
import tifffile
import warnings
import numpy as np

from tqdm import tqdm
from image_cut import create_restriction, do_cut, get_rand_centers
from torchvision import io
from torchvision.transforms.functional import resize

warnings.filterwarnings(action='ignore')

ROOT = '../data/OSCD/raw'
DST = '../data/OSCD/annotated'
W, H = (785, 799)
PATCH_SIZE = (256, 256)
N_PIX = 0
TRUE_PIX = 0

if not os.path.isdir(DST):
    os.mkdir(DST)


def read_dirs(path):
    subs_list = os.listdir(path)

    return subs_list


def img_paths(dirs):
    file_paths = {}

    for dir in dirs:
        path = os.path.join(ROOT, dir)
        cm = os.path.join(path, 'cm')
        pair = os.path.join(path, 'pair')

        cm_path = os.path.join(cm, 'cm.png')
        img1_path = os.path.join(pair, 'img1.png')
        img2_path = os.path.join(pair, 'img2.png')

        file_paths[dir] = {'cm': cm_path,
                           'img1': img1_path,
                           'img2': img2_path}

    return file_paths


def read_img_trio(paths):
    img1 = io.read_image(paths['img1'])
    img2 = io.read_image(paths['img2'])
    cm = io.read_image(paths['cm'], io.ImageReadMode.GRAY)

    s = cm.shape
    t_pix = np.prod(s)
    true_pix = cm.sum()

    I = resize(torch.cat((img1, img2, cm), dim=0), size=[W, H])

    return I, (t_pix, true_pix)


def create_dataset(paths, c):
    image, dtl = read_img_trio(paths)

    return do_cut(image.numpy(), c, PATCH_SIZE), dtl


if __name__ == '__main__':
    img_dict = img_paths(read_dirs(ROOT))

    for dir, dic in tqdm(img_dict.items(), desc='raw image trios: '):
        anchors = get_rand_centers(create_restriction(W, H, 100, PATCH_SIZE), 30)

        for i, a in enumerate(anchors):
            cut, _ = create_dataset(dic, a)
            dst_path = os.path.join(DST, f'{dir}_{i}.tiff')
            tifffile.imwrite(dst_path, cut)
