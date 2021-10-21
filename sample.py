import argparse
from glob import glob
from pathlib import Path
import os

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--sample-type', type=str, help='random')
    

    args = parser.parse_args()
    if args.dst:
        checkpath(args.dst)

    print(args)

    return args


if __name__ == '__main__':
    args = parse_args()
    for dirpath, fpaths in walk(args.imgpath, regex=args.regex):
        samples = [fpaths[ind] for ind in range(0, len(fpaths), 100)]
        for sample in samples:
            sample_path = Path(sample).parent.relative_to(args.imgpath)
            checkpath(args.dst / sample_path, ok='exist_ok')

            os.system('cp -r {} {}'.format(sample, args.dst/sample_path))
