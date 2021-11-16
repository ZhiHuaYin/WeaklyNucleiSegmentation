import os
import random
from tqdm import tqdm
from functools import partial

import argparse
import cv2
import numpy as np
from skimage.io import imread, imsave, imread_collection
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import label
from skimage.morphology import skeletonize
from multiprocessing import Pool

import warnings


def scribblize(mask, ratio=0.1):
    def remove_corner(mask, coords):
        for coord in coords:
            x, y = coord
            mask[x - 1][y - 1] = 0
            mask[x - 1][y] = 0
            mask[x - 1][y + 1] = 0
            mask[x][y - 1] = 0
            mask[x][y] = 0
            mask[x][y + 1] = 0
            mask[x + 1][y - 1] = 0
            mask[x + 1][y] = 0
            mask[x + 1][y + 1] = 0
        return mask

    """
    Automatically generate scribble-label with a specific ratio.
    the number of foreground scribbles is same with the number of background scribbles
    :param mask: fully annotated label
    :param ratio: scribble ratio
    :return: foreground & background scribbles
    """
    sk = skeletonize(mask)

    i_mask = np.abs(mask - 1) // 255
    i_sk = skeletonize(i_mask)
    coords = corner_peaks(corner_harris(i_sk), min_distance=5)
    i_sk = remove_corner(i_sk, coords)

    label_sk = label(sk)
    n_sk = np.max(label_sk)
    n_remove = int(n_sk * (1 - ratio))
    removes = random.sample(range(1, n_sk + 1), n_remove)
    for i in removes:
        label_sk[label_sk == i] = 0
    sk = (label_sk > 0).astype('uint8')

    label_i_sk = label(i_sk)
    n_i_sk = np.max(label_i_sk)
    n_i_remove = n_i_sk - (n_sk - n_remove)
    removes = random.sample(range(1, n_i_sk + 1), n_i_remove)
    for i in removes:
        label_i_sk[label_i_sk == i] = 0
    i_sk = (label_i_sk > 0).astype('uint8')
    return sk, i_sk


def process(label_path, ratio=0.1, shape=None):
    label = cv2.imread(label_path, flags=0)
    if shape[0] is not None and shape[1] is not None:
        label = cv2.resize(label, shape)

    mask = (label > 0).astype('uint8')
    sk, i_sk = scribblize(mask, ratio=ratio)
    scr = np.ones_like(mask) * 250
    scr[i_sk == 1] = 0
    scr[sk == 1] = 1
    return scr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    # parser.add_argument('--width', type=int, default=1440)
    # parser.add_argument('--height', type=int, default=992)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def work(file, root, output_path, args):
    if file is not None:
        if not os.path.exists(os.path.join(output_path, file)):
            result = process(os.path.join(root, file), ratio=args.ratio, shape=(args.width, args.height))
            imsave(os.path.join(output_path, file), result)


def main():
    warnings.filterwarnings("ignore")

    args = parse_args()
    output_path = os.path.join(args.output, f'scribble{int(100 * args.ratio)}')
    os.makedirs(output_path, exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(args.input):
        file_list.extend(files)

    random.shuffle(file_list)

    if args.j == 0:
        for file in file_list:
            work(file, root=args.input, output_path=output_path, args=args)
        return

    if len(file_list) % args.j != 0:
        file_list.extend([None] * (args.j - len(file_list) % args.j))

    with Pool(args.j) as p:
        list(tqdm(
            p.imap_unordered(partial(work, root=args.input, output_path=output_path, args=args), file_list),
            total=len(file_list), ascii=True
        ))
        p.close()


if __name__ == '__main__':
    main()
