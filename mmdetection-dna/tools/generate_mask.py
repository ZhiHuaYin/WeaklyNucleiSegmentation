import argparse
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from mmdet.datasets.api_wrappers import COCO

output_dirs = ['mask', 'image_visual']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str)
    parser.add_argument('--image-root', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    for d in output_dirs:
        os.makedirs(os.path.join(args.output, d), exist_ok=True)
    print(f'Output Dir:  {args.output}')

    coco = COCO(args.label)
    img_ids = coco.getImgIds()

    for idx in tqdm(range(len(img_ids)), ascii=True):
        img_id = img_ids[idx]
        json_img = coco.imgs[img_id]
        json_ann = coco.imgToAnns[img_id]

        img_path = os.path.join(args.image_root, json_img['file_name'])
        img = cv2.imread(img_path)
        mask = np.zeros((img.shape[0], img.shape[1]))

        contours = []
        for x in json_ann:
            seg = x['segmentation'][0]
            contour = np.array(seg).reshape((-1, 1, 2))
            contours.append(contour)

        img_vis = cv2.imread(img_path)
        img_vis = cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 5)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

        file_name = os.path.basename(img_path)
        file_name = file_name.replace('.jpg', '.png')

        cv2.imwrite(os.path.join(args.output, output_dirs[0], file_name), mask)
        cv2.imwrite(os.path.join(args.output, output_dirs[1], file_name), img_vis)


if __name__ == '__main__':
    main()
