import argparse
import json
import os
from collections import defaultdict

import numpy as np
import tqdm
from PIL import Image
from pycocotools import mask as _mask
from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ann', type=str, default='../data/coco_dna/annotations_new/test.json')
    parser.add_argument('--image-root', type=str, default='../data/coco_dna/image/tt/')
    parser.add_argument('--thresh', type=float, default=0.0)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def process(x, images, output_dir):
    img_id, pred = x

    mask = None
    for x in pred:
        m = _mask.decode(x['segmentation'])
        if mask is None:
            mask = m
        else:
            mask = mask | m
    if mask is None:
        mask = np.zeros((2816, 4096))
    mask = mask.astype(np.uint8)
    filename = os.path.basename(images[img_id]).replace('.jpg', '.png')
    Image.fromarray(mask * 255).save(os.path.join(output_dir, filename))


if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.join(os.path.dirname(args.pred), f'pred_{args.thresh}')
    os.makedirs(output_dir, exist_ok=True)

    with open(args.ann) as f:
        label = json.load(f)

    with open(args.pred) as f:
        pred_label = json.load(f)

    results = defaultdict(list)
    for det in pred_label:
        if det['score'] > args.thresh:
            results[det['image_id']].append(det)

    images = {}
    img_ids = []
    for x in label['images']:
        images[x['id']] = x['file_name']
        img_ids.append(x['id'])

    if args.j == 0:
        for img_id in tqdm.tqdm(img_ids, ascii=True):
            pred = results[img_id]

            mask = None
            for x in pred:
                m = _mask.decode(x['segmentation'])
                if mask is None:
                    mask = m
                else:
                    mask = mask | m
            if mask is None:
                mask = np.zeros((2816, 4096))
            mask = mask.astype(np.uint8)
            filename = os.path.basename(images[img_id]).replace('.jpg', '.png')
            Image.fromarray(mask * 255).save(os.path.join(output_dir, filename))
    else:
        result_list = []
        for img_id in img_ids:
            result_list.append((img_id, results[img_id]))

        with Pool(args.j) as p:
            list(tqdm.tqdm(
                p.imap(partial(process, images=images, output_dir=output_dir), result_list),
                total=len(result_list), ascii=True
            ))
            p.close()
