import os
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
from dataset.iterator import DetRecordIter
from config.cfg import cfg
from utils.metric import CornerNetMetric
import argparse
import logging
from symbols.cornernet import CornerNet
from utils.decode import decode

def parse_arg():
    parser = argparse.ArgumentParser(description='Train cornernet detection network')
    args = parser.parse_args()
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data/coco', 'val.rec'), type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default=os.path.join(os.getcwd(), 'data/coco', 'val.lst'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=100, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'exp1', 'cornerNet'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='1', type=str)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=511,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=500,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--num-class', dest='num_class', type=int, default=80,
                        help='number of classes')
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
    ctx_str = '(' + ','.join([str(c) for c in ctx]) + ')'
    cfg['network']['num_ctx'] = len(ctx)

    sym = CornerNet(is_train = False, cfgs = cfg['network'])

    if isinstance(args.data_shape, int):
        data_shape = (3,args.data_shape, args.data_shape)
    mean_pixels = [args.mean_r,args.mean_g,args.mean_b]
    if not args.DEBUG:
        eval_iter = DetRecordIter(cfg['network'], args.val_path, cfg['network']['batch_size'], data_shape, mean_pixels = mean_pixels,label_pad_width = args.label_width, path_imglist = args.val_list)
    else:
        eval_iter = DetRecordIter(cfg['network'], 'data/mini_train.rec', cfg['network']['batch_size'], data_shape, mean_pixels = mean_pixels,label_pad_width = args.label_width, path_imglist = 'data/mini_train.lst')

    mod = mx.mod.Module(symbol = sym, context = ctx, data_names = ['data'], label_names = eval_iter.label_names)
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    _, params, auxs = mx.model.load_checkpoint(args.prefix,args.epoch)
    mod.set_params(params, auxs, allow_missing=False, force_init=True)

    eval_iter.reset()
    for it in eval_iter:
        mod.forward(it,is_train = False)
        out = mod.get_outputs()
        detections = decode(*out[1:])
        import pdb
        pdb.set_trace()
        xxx = 1


if __name__ == '__main__':
    args = parse_arg()
    print(args)
    evaluate_net(args)
