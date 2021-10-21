import re

import numpy as np


def usort(fnames):
    if isinstance(fnames, dict):
        fnames = dict(sorted(fnames.items(), key=lambda k: int(re.sub(r'[^0-9]', '', k[0]))))
    elif isinstance(fnames, list):
        if len(fnames) and re.sub(r'[^0-9]', '', fnames[0]) != '':
            if isinstance(fnames[0], list):
                fnames = sorted(fnames, key=lambda k:int(re.sub(r'[^0-9]', '', k[0])))
            elif isinstance(fnames[0], dict):
                fnames = dict(sorted(fnames.items(), key=lambda k:int(re.sub(r'[^0-9]', '', k))))
            else:
                fnames = sorted(fnames, key=lambda fname:int(re.sub(r'[^0-9]', '', fname)))
            return fnames
    else:
        pass
    return fnames

def clip(x, xmin, xmax):
    if x <= xmin:
        return xmin
    elif x >= xmax:
        return xmax
    else:
        return x

def MergeImgs(imgs):
    if len(imgs) == 1:
        imgshow = imgs[0]
    elif len(imgs) == 2:
        imgshow = np.hstack([imgs[0], imgs[1]])
    elif len(imgs) == 3:
        imgshow = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], np.zeros_like(imgs[0])])])
    elif len(imgs) == 4:
        imgshow = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], imgs[3]])])
    else:
        assert(False), 'imgs number:{} > 4'.format(len(imgs))
    
    return imgshow
