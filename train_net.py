import argparse
import mxnet as mx
import os
import sys
from config.cfg import cfg
import logging
from dataset.iterator import DetRecordIter
from symbols.cornernet import CornerNet
import numpy as np
DEBUG = True
COCO2014 = True
# if COCO2014:
#   coco_dict =dict(zip(range(81), ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']))
# else:
coco_dict ={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
import time
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    if not DEBUG: 
        parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data/coco_rec', 'train.rec'), type=str)
    else:
        parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data/coco_rec', 'val.rec'), type=str)

    parser.add_argument('--train-list', dest='train_list', help='train list to use',
                        default="", type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data/coco_rec', 'val.rec'), type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'exp1', 'cornerNet'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=240, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=511,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=1000,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80, 160',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--num-example', dest='num_example', type=int, default=16551,
                        help='number of image examples')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=400,
                        help='final number of detections')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    parser.add_argument('--tensorboard', dest='tensorboard', type=bool, default=False,
                        help='save metrics into tensorboard readable files')
    parser.add_argument('--min_neg_samples', dest='min_neg_samples', type=int, default=0,
                        help='min number of negative samples taken in hard mining.')

    args = parser.parse_args()
    return args
def parse_class_names(args):
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

def train_net(args):
    prefix = args.prefix
    if os.path.exists(args.train_path.replace('rec','idx')):
        with open(args.train_path.replace('rec','idx'),'r') as f:
            txt = f.readlines()
        num_example = len(txt)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.log_file:
        log_file_path = os.path.join(os.path.dirname(prefix),args.log_file)
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)

    if isinstance(args.data_shape, int):
        data_shape = (3,args.data_shape, args.data_shape)

    if prefix.endswith('_'):
        prefix += '_' +str(data_shape[1])

    sym = CornerNet(is_train = True, cfgs = cfg['network'])
    mean_pixels = [args.mean_r,args.mean_g,args.mean_b]

    train_iter = DetRecordIter(cfg['network'], args.train_path, cfg['network']['batch_size'], data_shape, mean_pixels = mean_pixels,
                               label_pad_width = args.label_width, path_imglist = args.train_list)
    train_iter.reset()
    if DEBUG:
        it = train_iter.next()

        
        import cv2
        imgs = it.data[0].asnumpy().transpose(0,2,3,1)
        imgs *=70
        imgs += 110
        i = 0
        labels = it.label
        r = 511 / 128
        for img in imgs:
            img = img[:,:,::-1].copy()
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

            # cv2.imshow("img".format(i),img)
            # cv2.waitKey()

        mod = mx.mod.Module(symbol = sym, context = mx.gpu(0), data_names = ['data'], label_names = train_iter.label_names)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Uniform(scale=.1))
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
        mod.forward(it, is_train=True) 
        print(mod.get_outputs())
        a = 1
        mod.backward()
        print(mod._exec_group.grad_arrays)
        b = 1

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    class_names = parse_class_names(args)

    train_net(args)
