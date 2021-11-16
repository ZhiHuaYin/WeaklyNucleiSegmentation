import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    file_list = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            file_list.append(file.replace('.png', ''))

    output = {
        'ImageID': file_list,
        'fold': [args.fold] * len(file_list)
    }

    df = pd.DataFrame(output)
    df.to_csv(args.output, index=False)

    print(df)
    print(f'save csv to {args.output}')


if __name__ == '__main__':
    main()
