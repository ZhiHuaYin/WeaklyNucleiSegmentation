import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from skimage import io, measure
from tqdm import tqdm
import cv2


def get_mask_regions(label_mask):
    image_label = measure.label(label_mask)
    image_regions = measure.regionprops(image_label)
    return image_regions


def get_label_point(image_shape, label_regions):
    label_regions_center = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for region in label_regions:
        box_center = region.centroid
        cent_y = int(box_center[0])
        cent_x = int(box_center[1])
        label_regions_center[
            cent_y,
            cent_x
        ] = 255
    return label_regions_center


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--width', type=int, default=0)
    # parser.add_argument('-height', type=int, default=992)
    # parser.add_argument('-width', type=int, default=1440)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def work(file, root, args):
    if file is not None:
        if not os.path.exists(os.path.join(args.output, file)):
            # CODE HERE
            # label = io.imread(os.path.join(root, file))
            # label_regions = get_mask_regions(label)
            # label_regions_center = get_label_point(label.shape, label_regions)
            # io.imsave(os.path.join(args.output, file), label_regions_center, check_contrast=False)

            label = cv2.imread(os.path.join(root, file), flags=0)
            if args.height > 0 and args.width > 0:
                label = cv2.resize(label, (args.width, args.height))
            label[label > 0] = 255

            label_regions = get_mask_regions(label)
            label_regions_center = get_label_point(label.shape, label_regions)
            cv2.imwrite(os.path.join(args.output, file), label_regions_center)


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(args.input):
        file_list.extend(files)

    if args.j == 0:
        for file in file_list:
            work(file, root=args.input, args=args)
        return

    if len(file_list) % args.j != 0:
        file_list.extend([None] * (args.j - len(file_list) % args.j))

    with Pool(args.j) as p:
        list(tqdm(
            p.imap_unordered(partial(work, root=args.input, args=args), file_list),
            total=len(file_list), ascii=True
        ))
        p.close()


if __name__ == '__main__':
    main()
