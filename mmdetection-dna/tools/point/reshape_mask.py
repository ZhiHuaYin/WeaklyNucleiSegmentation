import argparse
import os
from functools import partial
from multiprocessing import Pool

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--height', type=int, default=992)
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def work(file, root, args):
    if file is not None:
        if not os.path.exists(os.path.join(args.output, file)):
            # CODE HERE
            image = cv2.imread(os.path.join(root, file), flags=0)
            image = cv2.resize(image, (args.width, args.height))
            image[image > 0] = 255
            cv2.imwrite(os.path.join(args.output, file), image)


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
