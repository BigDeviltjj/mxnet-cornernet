import mxnet as mx
import numpy as np

def conv(inputs, k, out_dim, stride, is_train, name, with_bn = True):
    p = (k - 1)//2
    conv = mx.symbol.Convolution(data = inputs, num_filter = out_dim, kernel = (k,k), pad = (p, p), stride = (stride, stride), no_bias = with_bn, name = name + '_conv')
    if with_bn:
        conv = mx.symbol.contrib.SyncBatchNorm(name = name +'_bn', data = conv, use_global_stats = not is_train, fix_gamma = False)
    relu = mx.symbol.Activation(name = name + '_relu', data = conv, act_type = 'relu')
    return relu

def residual(inputs, out_dim, stride, is_train, name):
    conv1 = conv(inputs, 3, out_dim, stride = stride, is_train = is_train, name = name+'_res1')
    conv2 = mx.symbol.Convolution(data = conv1, num_filter = out_dim, kernel = (3,3), pad = (1, 1), stride = (1, 1), no_bias = True, name = name + '_res2_conv')
    bn2 = mx.symbol.contrib.SyncBatchNorm(name = name +'_res2_bn', data = conv2, use_global_stats = not is_train, fix_gamma = False)
    relu2 = mx.symbol.Activation(name = name + '_res2_relu', data = bn2, act_type = 'relu') 

    skip = mx.symbol.Convolution(name= name + '_res_skip_conv', data=inputs, num_filter=out_dim, pad=(0, 0), kernel=(1, 1), stride=(stride, stride), no_bias=True)
    skip = mx.symbol.contrib.SyncBatchNorm(name = name + '_res_skip_bn', data=skip, use_global_stats=not is_train, fix_gamma=False)
    res = mx.symbol.broadcast_add(name= name + '_res_add', *[relu2, skip])
    res_relu = mx.symbol.Activation(name = name + '_res_out', data = res, act_type = 'relu')

    return res_relu
    


def pre(inputs, is_train):
    pre_conv = conv(inputs, 7, 128, 2, is_train, 'pre')
    pre_residual = residual(pre_conv,  256,2, is_train, 'pre')
    return pre_residual
def Hourglass(inputs, is_train):
    return inputs

def CornerNet(is_train, cfgs):
    data = mx.sym.Variable('data')
    inter = pre(data, is_train)
    feat = Hourglass(inter, is_train)
    return feat


if __name__ == '__main__':
    if DEBUG:
        out = CornerNet(True, None)
        net_visualization(network='', num_classes = 80, train=True, output_dir = './', print_net = True, net = out)
