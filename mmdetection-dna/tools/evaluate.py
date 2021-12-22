import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', type=str)
    parser.add_argument('pred', type=str)
    parser.add_argument('-j', type=int, default=0)

    return parser.parse_args()


def evaluate(pred, label):
    if label.shape != pred.shape:
        pred = cv2.resize(pred, (label.shape[1], label.shape[0]))

    pred = pred > 127
    label = label > 0

    tp = np.sum(label & pred)
    p = np.sum(pred)
    fn = np.sum(label) - tp
    return tp, p, fn


def process(filename, args):
    pred = cv2.imread(os.path.join(args.pred, filename), flags=0)
    mask = cv2.imread(os.path.join(args.label, filename), flags=0)

    return evaluate(pred, mask)


if __name__ == '__main__':
    args = parse_args()

    total_tp, total_p, total_fn = 0, 0, 0

    if args.j == 0:
        for i, file in enumerate(tqdm(os.listdir(args.pred), ascii=True)):
            pred = cv2.imread(os.path.join(args.pred, file), flags=0)
            mask = cv2.imread(os.path.join(args.label, file), flags=0)

            tp, p, fn = evaluate(pred, mask)
            total_tp += tp
            total_p += p
            total_fn += fn
    else:
        file_list = os.listdir(args.pred)
        with Pool(args.j) as p:
            output = list(tqdm(
                p.imap(partial(process, args=args), file_list),
                total=len(file_list), ascii=True
            ))
            p.close()

        for x in output:
            tp, p, fn = x
            total_tp += tp
            total_p += p
            total_fn += fn

    iou = total_tp / (total_p + total_fn)
    dice = 2 * total_tp / (total_p + total_tp + total_fn)
    ppv = total_tp / total_p
    s = total_tp / (total_tp + total_fn)
    print(f'iou: {iou:.5f}  dice: {dice:.5f}  ppv: {ppv:.5f}  s:{s:.5f}')
    print(f'{iou * 100:.3f} {dice * 100:.3f} {ppv * 100:.3f} {s * 100:.3f}')
