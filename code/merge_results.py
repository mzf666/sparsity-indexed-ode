import argparse
import os
import socket

import pandas as pd
from PyPDF2 import PdfFileMerger


def build_args():
    HOST_NAME = socket.gethostname()
    DATA_DICT = {'sparse-docker': '/HappyResearch/Data', }
    MODEL_DICT = {'sparse-docker': "/HappyResearch/Models", }
    OUT_DICT = {'sparse-docker': "/HappyResearch/Results", }

    parser = argparse.ArgumentParser('Merging results .csv / .pdf to target dir.')

    parser.add_argument('--out_dir', type=str, default=OUT_DICT[HOST_NAME])
    parser.add_argument('--dataset', type=str, default='cifar10 | cifar100')
    parser.add_argument('--exp', type=str, default='oneshot | iter')

    # parser.add_argument('--from_dir', type=str, default='the experiment dir')
    # parser.add_argument('--to_dir', type=str, default='the result dir')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--type', type=str, default='pdf', help='csv | pdf')

    args = parser.parse_args()

    args.from_dir = f'{args.out_dir}/{args.exp}_{args.dataset}'
    args.to_dir = args.out_dir
    print(f'From dir = {args.from_dir}')
    print(f'To dir = {args.to_dir}')

    args.to_dir = args.to_dir or args.from_dir
    args.from_file = args.from_dir.split('/')[-1]
    args.to_file = args.to_dir.split('/')[-1]
    args.title = args.title or args.from_file

    return args


def merge_pdf(from_dir, title, to_dir):
    files = [file for file in os.listdir(from_dir) if file.endswith('.pdf')]
    files = sorted(files)
    sheet_names = [file.rstrip('.pdf').rstrip('_summary') for file in files]
    paths = [f'{args.from_dir}/{file}' for file in files]

    to_path = f'{to_dir}/{title}.pdf'
    with PdfFileMerger() as merger:
        for name, path in zip(sheet_names, paths):
            merger.append(path, name)

        merger.write(to_path)


def merge_csv(from_dir, title, to_dir):
    files = [file for file in os.listdir(from_dir) if file.endswith('.csv')]
    files = sorted(files)
    sheet_names = [file.rstrip('.csv').rstrip('_summary') for file in files]
    paths = [f'{args.from_dir}/{file}' for file in files]

    to_path = f'{to_dir}/{title}.xlsx'
    with pd.ExcelWriter(to_path) as writer:
        for name, path in zip(sheet_names, paths):
            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=name, index=False)

        writer.save()


if __name__ == '__main__':
    args = build_args()

    if args.type == 'csv':
        merge_csv(args.from_dir, args.title, args.to_dir)
        print(f'\n.csv in \n{args.from_dir}\n merged successfully as {args.title}.xlsx to \n{args.to_dir}.\n')

    elif args.type == 'pdf':
        merge_pdf(args.from_dir, args.title, args.to_dir)
        print(f'\n.pdf in \n{args.from_dir}\n merged successfully as {args.title}.pdf to \n{args.to_dir}.\n')
