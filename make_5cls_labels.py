import glob
from pathlib import Path
import os
from utils.io import annwrite

from utils import *

src = '/data1/Personal/linaifan/nm_train/slabels_norm'
dst = '/data1/Personal/linaifan/nm_train_5cls/slabels_norm_5cls'
fpaths = glob.glob('{}/**/*txt'.format(src), recursive=True)
for fpath in fpaths:
    with open(fpath, 'r') as fr:
        anns_tmp = fr.readlines()
        anns_tmp = [ann.strip() for ann in anns_tmp if ann.strip()]
        anns = []
        for ann in anns_tmp:
            ann = ann.split()
            if ann[0] in ['0', '1', '2', '3']:
                anns.append(ann)
            elif ann[0] in ['4', '5', '6', '7']:
                ann[0] = '4'
                anns.append(ann)
            else:
                continue

    fpath = Path(fpath)
    rel_fpath = fpath.relative_to(src)
    dst_fpath = dst / rel_fpath
    checkpath(dst_fpath.parent, ok='exist_ok')
    print(dst_fpath)

    if len(anns):
        annwrite(dst_fpath, anns)
    else:
        os.system('touch {}'.format(dst_fpath))
    
    

    