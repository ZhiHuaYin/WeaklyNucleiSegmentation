import argparse
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from mmdet.datasets.api_wrappers import COCO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', type=str)
    parser.add_argument('image_root', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--reshape', type=bool, default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

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

        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

        if args.reshape:
            mask = cv2.resize(mask, (1440, 992))
            mask[mask > 0] = 255

        file_name = os.path.basename(img_path)
        file_name = file_name.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(args.output, file_name), mask)


if __name__ == '__main__':
    main()
