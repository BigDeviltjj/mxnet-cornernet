import os
import pprint
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
from dataset.testloader import TestDetRecordIter
from dataset.load_data import load_gt_roidb, merge_roidb, filter_roidb
from config.cfg import cfg
from utils.metric import CornerNetMetric
import argparse
import logging
from symbols.cornernet import CornerNet
from utils.decode import decode
import numpy as np
import cv2
from dataset.dataset import coco
coco_dict ={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def parse_arg():
    parser = argparse.ArgumentParser(description='Train cornernet detection network(will only use one gpu)')
    parser.add_argument('--val-set', dest='valset', help='validation record to use',
                        default='minival2014', type=str)
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='coco', type=str)
    parser.add_argument('--root-dir', dest='root_dir', help='root dir of data file to use',
                        default='./data', type=str)
    parser.add_argument('--image-dir', dest='image_dir', help='image dir of data file to use',
                        default='./data/coco', type=str)
    parser.add_argument('--result-path', dest='result_path', help='dir of result',
                        default='./output', type=str)

    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=100, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'exp1', 'cornerNet'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='1', type=str)
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=100,
                        help='final number of detections')
    parser.add_argument('--debug', dest='DEBUG', type=bool, default=False,
                        help='debug mode')
    args = parser.parse_args()


    return args

def evaluate_net(args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    ctx = ctx[0]
    cfg['num_ctx'] = cfg['batch_size']

    sym = CornerNet(is_train = False, cfgs = cfg)

    valset = args.valset

    imdb = eval(args.dataset)(valset, args.root_dir, args.image_dir, args.result_path)
    val_roidb = imdb.gt_roidb()
    if args.DEBUG:
        val_roidb = val_roidb[0:100]

    eval_iter = TestDetRecordIter(val_roidb, cfg, 1,False)

    if args.DEBUG and 0:
        
        mod = mx.mod.Module(symbol = sym, context = ctx, data_names = ['data'])
        mod.bind(data_shapes=[('data',(2,3,511,767))], label_shapes=None)
        _, params, auxs = mx.model.load_checkpoint('model/pretrained_cornernet',0)
        mod.set_params(params, auxs, allow_missing=False, force_init=True)
        it = np.load('image.npy')
        mod.forward(mx.io.DataBatch(data=[mx.nd.array(it)]),is_train = False)
        out = mod.get_outputs()[0]
    mod = mx.mod.Module(symbol = sym, context = ctx, data_names = eval_iter.data_names, label_names = None)
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=None, for_training = False)
    _, params, auxs = mx.model.load_checkpoint(args.prefix,args.epoch)
    mod.set_params(params, auxs, allow_missing=False, force_init=True)


    num_classes = cfg['num_classes']
    all_boxes = [[[] for _ in range(len(val_roidb))] for _ in range(num_classes + 1)]
    eval_iter.reset()
    for idx,(info, it) in enumerate(eval_iter):
        print('predicting image %d'%idx)
        mod.forward(it,is_train = False)
        out = mod.get_outputs()
        bboxes = decode(*out, info, cfg['test_scales'])
        
        for i in range(num_classes):
            all_boxes[i + 1][idx] = bboxes[i+1]
        if args.DEBUG:

            for i in range(1):
                im = it.data[0].asnumpy()
                im = im[i].transpose((1,2,0))
                im *= 70
                im += 110
                im_map = im.copy()
                im = im.clip(0,255).astype(np.uint8)
                h,w,c = im.shape
                
                tl_map = mx.nd.sigmoid(out[0]).asnumpy()
                tl_map = tl_map[i].transpose(1,2,0).sum(axis = -1) * 255
                tl_map = cv2.resize(tl_map,(w,h))
                br_map = mx.nd.sigmoid(out[1]).asnumpy()
                br_map = br_map[i].transpose(1,2,0).sum(axis = -1) * 255
                br_map = cv2.resize(br_map,(w,h))
                im_map[:,:,0] += br_map
                im_map[:,:,-1] += tl_map
                cv2.imwrite('images/heat_%04d.jpg'%idx, im_map)
                
                
                for i in range(1, 81):
                    cat_name = coco_dict[i-1]
                    cat_size = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    color = np.random.random((3,)) * 0.6 + 0.4
                    color = (color * 255).astype(np.int32).tolist()
                    for bbox in bboxes[i]:
                        b = bbox[:4]
                        b[0:4:2] += info[0][0,2]
                        b[1:4:2] += info[0][0,0]
                        b = b.astype(np.int)
                        if bbox[-1] <0.5:
                            continue
                        if b[1] - cat_size[1] - 2< 0:
                            cv2.rectangle(im, (b[0], b[1]+2),
                                              (b[0] + cat_size[0], b[1]+cat_size[1] + 2),
                                              color, -1)
                            cv2.putText(im, cat_name,(b[0],b[1]+cat_size[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
                        else:
                            cv2.rectangle(im, (b[0], b[1] - cat_size[1] - 2),
                                              (b[0] + cat_size[0], b[1] - 2),
                                              color, -1)
                            cv2.putText(im, cat_name,(b[0],b[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
                        cv2.rectangle(im, (b[0],b[1]),(b[2],b[3]),color,2)
                cv2.imwrite('images/image_%04d.jpg'%idx, im)
    if not args.DEBUG:
        info_str = imdb.evaluate_detections(all_boxes)
        pprint.pprint(info_str)
                    
            


if __name__ == '__main__':
    args = parse_arg()
    print(args)
    evaluate_net(args)
