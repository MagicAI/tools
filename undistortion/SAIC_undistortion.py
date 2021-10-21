#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

import sys
sys.path.insert(0, '..')
from utils import *
from pathlib import Path
import argparse

def LoadPinholeIntrinsic(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)

    fx = fs.getNode("fx").real()
    fy = fs.getNode("fy").real()
    cx = fs.getNode("cx").real()
    cy = fs.getNode("cy").real()

    k1 = fs.getNode("k1").real()
    k2 = fs.getNode("k2").real()

    p1 = fs.getNode("p1").real()
    p2 = fs.getNode("p2").real()

    k3 = fs.getNode("k3").real()
    k4 = fs.getNode("k4").real()
    k5 = fs.getNode("k5").real()
    k6 = fs.getNode("k6").real()

    K = np.array([[fx, 0, cx], 
                 [0, fy, cy], 
                 [0, 0, 1]])
    DIST_COEF = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    print(K)
    print(DIST_COEF)
    return K, DIST_COEF




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--ROOT', type=str)
    args = parser.parse_args()
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)

    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()

    intrinsic_path = 'SAIC_intrinsic'
    for dirpath, fpaths in walk(args.imgpath, regex=args.regex):
        for fpath in fpaths:
            if 'SideFrontLeft' in fpath:
                intrinsic_fpath = os.path.join(intrinsic_path, 'left_front', 'intrinsic.yaml')
            elif 'SideFrontRight' in fpath:
                intrinsic_fpath = os.path.join(intrinsic_path, 'right_front', 'intrinsic.yaml')
            elif 'SideRearLeft' in fpath:
                intrinsic_fpath = os.path.join(intrinsic_path, 'left_rear', 'intrinsic.yaml')
            elif 'SideRearRight' in fpath:
                intrinsic_fpath = os.path.join(intrinsic_path, 'right_rear', 'intrinsic.yaml')
            else:
                assert(False)
            print(intrinsic_fpath)
            K, DIST_COEF = LoadPinholeIntrinsic(intrinsic_fpath)
            distorted_image = cv2.imread(fpath)
            undistort_image= cv2.undistort(distorted_image, K, DIST_COEF)
            dst_imgfpath = args.dst / Path(fpath).relative_to(args.ROOT)
            dst_path = dst_imgfpath.parent
            if not dst_path.exists():
                dst_path.mkdir(parents=True)
                
            cv2.imwrite(dst_imgfpath.as_posix(), undistort_image)
