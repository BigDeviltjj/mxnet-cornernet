import numpy as np
from dataset.load_data import load_gt_roidb, merge_roidb, filter_roidb
from dataset.parallel_loader import DetRecordIter
from config.cfg import cfg
coco_dict ={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
image_sets = ['minival2014'] 
roidbs = [load_gt_roidb('coco',image_set,'./data','./data/coco') for image_set in image_sets]
roidb = merge_roidb(roidbs)
roidb = filter_roidb(roidb)
val_iter = DetRecordIter(roidb, cfg, cfg['batch_size'],False)
it = val_iter.next()
print(val_iter.data[0].shape)
import pdb
pdb.set_trace()

import cv2
imgs = it.data[0].asnumpy().transpose(0,2,3,1)
imgs *=70
imgs += 110
i = 0
labels = it.label
r = 511 / 128
for img in imgs:
    img = img[:,:,:].copy()
    tl_heat = labels[0][i].asnumpy()
    br_heat = labels[1][i].asnumpy()
    tl_reg = labels[2][i].asnumpy()
    br_reg = labels[3][i].asnumpy()
    tl_em = labels[4][i].asnumpy()#[i * cfg['network']['max_tag_len']:(i + 1) * cfg['network']['max_tag_len']]
    br_em = labels[5][i].asnumpy()#[i * cfg['network']['max_tag_len']:(i + 1) * cfg['network']['max_tag_len']]
    mask = labels[6][i].asnumpy()
    keep = np.where(mask == 1)[0]
    tl_reg = tl_reg[keep]
    br_reg = br_reg[keep]
    tl_em = tl_em[keep]
    br_em = br_em[keep]
    tl_h = (np.sum(tl_heat, axis = 0) * 255).astype(np.uint8)
    br_h = (np.sum(br_heat, axis = 0) * 255).astype(np.uint8)
    heat = cv2.resize((tl_h+br_h),(511,511))

    img += heat[:,:,np.newaxis]

    for tlr, brr, tle, bre in zip(tl_reg, br_reg, tl_em, br_em):
        tl = np.array([tle%128, tle/128],dtype = np.int)
        br = np.array([bre%128, bre/128],dtype = np.int)
        cls1 = np.where(tl_heat[:,tl[1],tl[0]] == 1.)[0]
        cls2 = np.where(br_heat[:,br[1],br[0]] == 1.)[0]
        cls = -1
        print(cls1, cls2)

        for c1 in cls1:
            for c2 in cls2:
                if c1 == c2:
                    cls = c1
                    break
        if cls == -1: assert 0, "no cls!!!"

        tl1 = tuple((tl * r).astype(int))
        br1 = tuple((br * r).astype(int))
        tl2 = tuple(((tl+tlr) * r).astype(int))
        br2 = tuple(((br+brr) * r).astype(int))
        print(tl1,br1)

        cv2.rectangle(img, tl1, br1, (0,0,255), 1)
        cv2.rectangle(img, tl2, br2, (255,0,0), 1)
        cv2.putText(img, str(coco_dict[cls1[0]]),(tl2[0],br2[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        print(str(coco_dict[cls]))
    print()
    img = img.clip(0,255).astype(np.uint8)
    i += 1

    cv2.imwrite("img+{}.jpg".format(i),img)

