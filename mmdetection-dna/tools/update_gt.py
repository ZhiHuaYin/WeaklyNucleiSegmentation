import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask
from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', default='../data/coco_dna/annotations/train_gray.json', type=str)
    parser.add_argument('pred', type=str)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def bbox_to_segm(box):
    x, y, h, w = box
    ann = []
    ann.extend([x, y])
    ann.extend([x + h // 2, y])
    ann.extend([x + h, y])
    ann.extend([x + h, y + w // 2])
    ann.extend([x + h, y + w])
    ann.extend([x + h // 2, y + w])
    ann.extend([x, y + w])
    ann.extend([x, y + w // 2])

    return [ann]


def mask2polygon(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if len(contour) > 4:
            contour_list = contour.flatten().tolist()
            segmentation.append(contour_list)
    return segmentation


def process(x):
    x['area'] = float(mask.area(x['segmentation']))
    x['segmentation'] = mask2polygon(mask.decode(x['segmentation']))
    if len(x['segmentation']) == 0:
        x['segmentation'] = bbox_to_segm(x['bbox'])
    x['iscrowd'] = 0
    return x


if __name__ == '__main__':
    args = parse_args()
    with open(args.label) as f:
        ann = json.load(f)

    with open(args.pred) as f:
        new_labels = json.load(f)

    print(len(ann['annotations']))
    print(len(new_labels))

    if args.j == 0:
        id = 0
        for x in tqdm(new_labels, ascii=True):
            x['area'] = float(mask.area(x['segmentation']))
            x['segmentation'] = mask2polygon(mask.decode(x['segmentation']))
            if len(x['segmentation']) == 0:
                x['segmentation'] = bbox_to_segm(x['bbox'])
            x['iscrowd'] = 0

            x['id'] = id
            id += 1

        ann['annotations'] = new_labels
    else:
        with Pool(args.j) as p:
            new_labels_out = list(tqdm(
                p.imap(partial(process), new_labels),
                total=len(new_labels), ascii=True
            ))
            p.close()

        id = 0
        for x in tqdm(new_labels_out, ascii=True):
            x['id'] = id
            id += 1

        ann['annotations'] = new_labels_out

    with open(args.pred.replace('segm.json', 'ann.json'), 'w') as f:
        json.dump(ann, f, indent=2)
