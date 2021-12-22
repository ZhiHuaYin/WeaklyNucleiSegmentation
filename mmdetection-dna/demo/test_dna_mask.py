# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='input dir')
    parser.add_argument('--output', help='output dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)

    os.makedirs(args.output, exist_ok=True)

    for root, dirs, files in os.walk(args.input):
        for file in tqdm(files):
            img = os.path.join(root, file)
            result = inference_detector(model, img)

            boxes = result[0][0]
            mask = None
            for i, m in enumerate(result[1][0]):
                if boxes[i][-1] < args.score_thr:
                    continue

                if mask is None:
                    mask = m
                else:
                    mask = mask | m
            if mask is None:
                image = cv2.imread(img, flags=0)
                mask = np.zeros(image.shape)
            mask = mask.astype(np.uint8)
            Image.fromarray(mask * 255).save(os.path.join(args.output, file.replace('.jpg', '.png')))


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
