import cv2
import random

from .cfgs import catenms_train, cate_colors

def draw_align_lines(img):
    imgh, imgw = img.shape[:-1]
    for lineh in range(0, imgh, 50):
        cv2.line(img, (0, lineh), (imgw-1, lineh), (0, 0, 255), thickness=1)

    cv2.line(img, (0, imgh//2), (imgw-1, imgh//2), (0, 255, 0), thickness=3)


def draw_bboxes(img, labels, imginfo='', show_conf=False, show_text=True):
    def compute_puttext_loc(imgw, basepoint, tsize, scale=1):
        xlt, ylt, xrb, yrb = basepoint
        if ylt - tsize[-1] - scale < 0:  # top truncated case
            ylt = max(ylt, 0) + tsize[-1] + (scale - 1)
        else:
            ylt -= (scale + 1)
        if xlt + tsize[0] > imgw:
            xlt = min(imgw, xrb) - tsize[0] - (scale)
        else:
            xlt -= (scale + 1)
            
        return (xlt, ylt)
    
    if len(labels) == 0:
        return img

    imgh, imgw = img.shape[:-1]

    if imgw == 1920:
        scale = 3
    elif imgw == 640:
        scale = 1
    elif imgw == 1280:
        scale = 2
    else:
        scale = 3

    for label in labels:
        cate, xlt, ylt, xrb, yrb = label['cate'], label['xlt'], label['ylt'], label['xrb'], label['yrb']
        xlt, ylt, xrb, yrb = map(lambda x: round(x), [xlt, ylt, xrb, yrb])
        #
        # if xlt <= 1 and ylt <= 1 and xrb <= 1 and yrb <= 1: # normalized xy
        #     h, w, _ = img.shape
        #     xlt, xrb = map(lambda x:x*w, [xlt, xrb])
        #     ylt, yrb = map(lambda x:x*h, [ylt, yrb])
        # xlt, ylt, xrb, yrb = map(int, [xlt, ylt, xrb, yrb])

        # draw bbox
        try:
            cate_color = cate_colors[int(cate)]
            cate = catenms_train[int(cate)]
        except:
            try:
                cate_color = cate_colors[catenms_train.index(cate)]
            except:
                cate_color = (0, 0, 255)
                print('cate: {} is not defined'.format(cate))

        cv2.rectangle(img, (xlt, ylt), (xrb, yrb), cate_color, thickness=scale)
        
        # putText
        if show_text:
            text = ''
            cate = cate[0] if cate in ['car', 'truck'] else cate[:3]
            if label['trackid']:
                text += label['trackid']
            text += cate
            if show_conf and label['conf']:
                text += str(round(label['conf'], 2))
            tsize, baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4*scale, thickness=scale)
            xlt, ylt = compute_puttext_loc(imgw, (xlt, ylt, xrb, yrb), tsize)
            cv2.rectangle(img, (xlt, ylt), (xlt + tsize[0], ylt - tsize[-1]), cate_color, -1)
            cv2.putText(img, text, (xlt, ylt), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale, (255, 255, 255), thickness=scale)

    if imginfo:
        cv2.putText(img, str(imginfo), (5*scale, 25*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.7*scale, (0, 0, 255), thickness=scale)

    return img
    