from __future__ import division
import argparse
import mxnet as mx
import os
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from config.cfg import cfg
import logging
from dataset.loader import DetRecordIter
from dataset.load_data import load_gt_roidb, merge_roidb, filter_roidb
from symbols.cornernet import CornerNet
import numpy as np
from utils.metric import CornerNetMetric
coco_dict ={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
import time
def parse_args():
    parser = argparse.ArgumentParser(description='Train cornernet detection network')
    parser.add_argument('--train-set', dest='trainset', help='train set to use',
                        default='train2014', type=str)
    parser.add_argument('--val-set', dest='valset', help='validation record to use',
                        default='minival2014', type=str)
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='coco', type=str)
    parser.add_argument('--root-dir', dest='root_dir', help='root dir of data file to use',
                        default='./data', type=str)
    parser.add_argument('--image-dir', dest='image_dir', help='image dir of data file to use',
                        default='./data/coco', type=str)
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default='model/pretrained_cornernet', type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'exp1', 'cornerNet'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=100, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=511,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=500,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.00025,
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
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='90',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--num-class', dest='num_class', type=int, default=80,
                        help='number of classes')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=100,
                        help='final number of detections')
    parser.add_argument('--checkpoint_period', dest='checkpoint_period', type=int, default=5,
                        help='checkpoint_period')
    parser.add_argument('--debug', dest='DEBUG', type=bool, default=False,
                        help='debug mode')

    args = parser.parse_args()
    return args

def get_optimizer_params(optimizer=None, learning_rate=None, momentum=None,
                         weight_decay=None, lr_scheduler=None, ctx=None, logger=None):
    if optimizer.lower() == 'rmsprop':
        opt = 'rmsprop'
        logger.info('you chose RMSProp, decreasing lr by a factor of 10')
        optimizer_params = {'learning_rate': learning_rate / 10.0,
                            'wd': weight_decay,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'sgd':
        opt = 'sgd'
        optimizer_params = {'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'adadelta':
        opt = 'adadelta'
        optimizer_params = {}
    elif optimizer.lower() == 'adam':
        opt = 'adam'
        optimizer_params = {'learning_rate': learning_rate,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    return opt, optimizer_params

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def train_net(args):
    DEBUG = args.DEBUG
    prefix = args.prefix
    num_example = 0

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

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    ctx_str = '(' + ','.join([str(c) for c in ctx]) + ')'
    cfg['num_ctx'] = len(ctx)
    sym = CornerNet(is_train = True, cfgs = cfg)
    mean_pixels = [args.mean_r,args.mean_g,args.mean_b]
    trainset = args.trainset.split('+')
    valset = args.valset.split('+')

    if not DEBUG:
        train_roidbs = [load_gt_roidb(args.dataset,image_set,args.root_dir,args.image_dir) for image_set in trainset]
        train_roidb = merge_roidb(train_roidbs)
        train_roidb = filter_roidb(train_roidb)

        val_roidbs = [load_gt_roidb(args.dataset,image_set,args.root_dir,args.image_dir) for image_set in valset]
        val_roidb = merge_roidb(val_roidbs)
        val_roidb = filter_roidb(val_roidb)

        train_iter = DetRecordIter(train_roidb, cfg, cfg['batch_size'],True)
        val_iter = DetRecordIter(val_roidb, cfg, cfg['batch_size'],False)
    else:

        val_roidbs = [load_gt_roidb(args.dataset,image_set,args.root_dir,args.image_dir) for image_set in valset]
        val_roidb = merge_roidb(val_roidbs)
        val_roidb = filter_roidb(val_roidb)
        val_roidb = val_roidb[4:8]
        train_roidb = val_roidb.copy()

        train_iter = DetRecordIter(train_roidb, cfg, cfg['batch_size'],False)
        val_iter = DetRecordIter(val_roidb, cfg, cfg['batch_size'],False)
    train_iter.reset()
    num_example = len(train_roidb)
    print('totally {} images'.format(num_example))
    if DEBUG and 0:
        import cv2
        it = train_iter.next()
        imgs = it.data[0].asnumpy().transpose(0,2,3,1)
        imgs *=70
        imgs += 110
        i = 0
        labels = it.label
        r = 511 / 128
        for img in imgs:
            img = img.copy()
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

            cv2.imwrite("images/image_test_{}.jpg".format(i),img)

        mod = mx.mod.Module(symbol = sym, context = ctx, data_names = ['data'], label_names = train_iter.label_names)
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Uniform(scale=.1))
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
        mod.forward(it, is_train=True) 
        print(mod.get_outputs())
        a = 1
        mod.backward()
        print(mod._exec_group.grad_arrays)
        b = 1
    params = None
    auxs = None
    fixed_param_names = None
    begin_epoch = args.begin_epoch
    if args.resume > 0:
        logger.info("Resume training with {} from epoch {}"
                    .format(ctx_str, resume))
        _, params, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif args.pretrained:
        logger.info("Start training with {} from pretrained model {}"
                    .format(ctx_str, args.pretrained))
        _,  params, auxs = mx.model.load_checkpoint(args.pretrained, args.epoch)
    else:
        logger.info("Experimental: start training from scratch with {}"
                    .format(ctx_str))
        params = None
        auxs = None
        fixed_param_names = None
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    mod = mx.mod.Module(sym, data_names = train_iter.data_names, label_names = train_iter.label_names, logger = logger, context = ctx, fixed_param_names = fixed_param_names)
    batch_end_callback = []
    eval_end_callback = []
    epoch_end_callback = [mx.callback.do_checkpoint(prefix, period = args.checkpoint_period)]
    batch_end_callback.append(mx.callback.Speedometer(train_iter.batch_size, frequent=args.frequent))

    learning_rate, lr_scheduler = get_lr_scheduler(args.learning_rate, args.lr_refactor_step, args.lr_refactor_ratio, num_example, train_iter.batch_size, begin_epoch)

    opt, opt_params = get_optimizer_params(optimizer = args.optimizer, learning_rate = learning_rate, momentum = args.momentum, weight_decay = args.weight_decay, lr_scheduler = lr_scheduler, ctx = ctx, logger = logger)

    valid_metric = CornerNetMetric() #TODO

    if DEBUG and 0:
        tic = time.time()
        train_iter.reset()
        n = 20
        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier())
        mod.init_optimizer(optimizer='adam', optimizer_params=opt_params)
        for i in range(n):
            it = train_iter.next()
            mod.forward_backward(it)
            mod.update()
            mod.update_metric(valid_metric, it.label)
        mx.nd.waitall()
        print(train_iter.batch_size*n/(time.time() - tic))
        assert 0,'finish'

    train_iter = mx.io.PrefetchingIter(train_iter)
    val_iter = mx.io.PrefetchingIter(val_iter)
    mod.fit(train_iter,
            val_iter,
            eval_metric=valid_metric,
            validation_metric=valid_metric,
            batch_end_callback=batch_end_callback,
            eval_end_callback=eval_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer=opt,
            optimizer_params=opt_params,
            begin_epoch=begin_epoch,
            num_epoch=args.end_epoch,
            initializer=mx.init.Xavier(),
            arg_params=params,
            aux_params=auxs,
            allow_missing=True)
if __name__ == '__main__':
    args = parse_args()

    train_net(args)
