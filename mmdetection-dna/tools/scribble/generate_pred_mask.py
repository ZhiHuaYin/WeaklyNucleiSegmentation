import argparse
import os

from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f'Output Dir:  {args.output}')

    for root, dirs, files in os.walk(args.input):
        for file in tqdm(files):
            mask = np.load(os.path.join(root, file))
            Image.fromarray((mask >= 0.5).astype(np.uint8) * 255).save(
                os.path.join(args.output, file.replace('.npy', '.png'))
            )


if __name__ == '__main__':
    main()
