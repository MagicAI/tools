#!/usr/bin/python
import argparse
from genericpath import exists, isfile
import os
import sys
from pathlib import Path
from utils.io import loadtxt

import cv2
import numpy as np

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpaths', nargs='+', type=str, required=True)
    parser.add_argument('--annpaths', nargs='+', type=str, default=None)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--classes', nargs='+', type=str, default=None)
    parser.add_argument('--align-lines', action='store_true')
    parser.add_argument('--save-type', type=str, help='txt/raw/merge')
    parser.add_argument('--view-4cam', action='store_true')
    parser.add_argument('--ROOT', type=str)
    args = parser.parse_args()
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)

    if args.view_4cam:
        assert(len(args.imgpaths) == 1)
        for cam in ['SideFrontRight', 'SideRearLeft', 'SideRearRight']:
            args.imgpaths.append(args.imgpaths[0].replace('SideFrontLeft', cam))

    if args.annpaths is not None and len(args.annpaths) == 1 and args.annpaths[0] == '-':
        args.annpaths = [imgpath.replace('image', 'label') for imgpath in args.imgpaths]


    classes = []
    if not args.classes:
        args.classes = catenms_train
    for cls in args.classes:
        classes.append(str(catenms_train.index(cls)))
    args.classes.extend(classes)

    print(args)

    return args

if __name__ == '__main__':
    args = parse_args()
    filters = loadtxt(args.filter) if args.filter else []

    imgpath0 = args.imgpaths[0]
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL) 

    for dirpath, fpaths in walk(imgpath0, regex=args.regex):
        idx = 0
        if filters: fpaths = [fpath for fpath in fpaths if fpath in filters]

        while 1:
            if idx >= len(fpaths): break
            imgfpath0 = Path(fpaths[idx])
            rel_imgfpath0 = imgfpath0.relative_to(imgpath0)
            
            imgs, det_imgs, imgfpaths = [], [], []
            # noshow_tag = False
            for i, imgpath in enumerate(args.imgpaths):                
                imgfpathi = imgpath / rel_imgfpath0
                if i == 0:
                    print('==> {}/{}, path: {}'.format(idx, len(fpaths)-1, imgfpathi))
                else:
                    print('    {}/{}, path: {}'.format(idx, len(fpaths)-1, imgfpathi))
                
                # change side imgfname
                # if args.side_cam:
                #     if i>=1 and not imgfpathi.exists():
                #         for j in range(10):
                #             imgfpathi_tmp = Path(imgfpathi.as_posix().replace('vc6', 'vc{}'.format(j)))
                #             if imgfpathi_tmp.exists():
                #                 imgfpathi = imgfpathi_tmp
                #                 break
                assert(imgfpathi.exists()), 'imgfpath: {}'.format(imgfpathi)
                img = cv2.imread(imgfpathi.as_posix())
                imgs.append(img.copy())
                # parse anns
                if args.annpaths is not None:
                    annfpath = Path(args.annpaths[i]) / rel_imgfpath0.with_suffix('.txt')
                    annfpath = Path(annfpath.as_posix().replace('image', 'label'))        
                    if not annfpath.exists():
                        print('    *** annfpath missing: {} ***'.format(annfpath))
                    
                    anns = parse_anns(annfpath.as_posix()) if annfpath.exists() else []

                    # classes filter
                    anns = [ann for ann in anns if ann['cate'] in args.classes]
                        # if i == 0 and len(anns) == 0:
                        #     noshow_tag = True
                        #     break
                    
                    # load image and draw 2D bbox
                    draw_bboxes(img, anns, imginfo=imgfpathi.name)
                
                # draw align lines
                if args.align_lines: draw_align_lines(img)
            
                det_imgs.append(img)
                imgfpaths.append(imgfpathi)

            # if noshow_tag:
            #     idx += 1
            #     continue

            imgshow = MergeImgs(det_imgs)
            cv2.imshow('viewimg', imgshow)
            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('j'):
                idx += 100
            elif key == ord('k'):
                idx -= 100
            elif key == ord('n'):
                idx += 5
            elif key == ord('m'):
                idx -= 5
            elif key == ord('b'):
                idx -= 1
            elif key == ord('s'):
                dst = args.dst if args.dst else 'tmp'
                if args.save_type == 'txt':
                    with open('{}/tmp.txt'.format(dst), 'a') as fa:
                        fa.write('{}\n'.format(imgfpath0))
                elif args.save_type == 'raw':
                    if args.view_4cam:
                        assert(args.ROOT)
                        for ind, imgfpath in enumerate(imgfpaths):
                            relfpath = dst / Path(imgfpath).relative_to(args.ROOT)
                            checkpath(relfpath.parent.as_posix(), ok='exist_ok')
                            cv2.imwrite('{}'.format(relfpath), imgs[ind])
                    else:
                        assert(args.ROOT)
                        for ind, imgfpath in enumerate(imgfpaths):
                            relfpath = dst / Path(imgfpath).relative_to(args.ROOT)
                            checkpath(relfpath.parent.as_posix(), ok='exist_ok')
                            cv2.imwrite('{}'.format(relfpath), imgs[ind])
                elif args.save_type == 'merge':
                    cv2.imwrite('{}/{}'.format(dst, rel_imgfpath0.with_suffix('.png').name), imgshow)
                else:
                    assert(False)
                idx += 1
            else:
                idx += 1

            if idx < 0:
                idx += len(fpaths) 

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows()        


