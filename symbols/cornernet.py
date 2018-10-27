import mxnet as mx
import numpy as np
import sys
sys.path.insert(0,'../')
print(__file__)
print(sys.path)
from py_operator.corner_pooling import *

DEBUG = True
if DEBUG:
    import sys
    from utils.visualize_net import net_visualization

def conv(inputs, k, out_dim, stride, is_train, name, with_bn = True):
    p = (k - 1)//2
    conv = mx.symbol.Convolution(data = inputs, num_filter = out_dim, kernel = (k,k), pad = (p, p), stride = (stride, stride), no_bias = with_bn, name = name + '_conv')
    if with_bn:
        conv = mx.symbol.BatchNorm(name = name +'_bn', data = conv, use_global_stats = not is_train, fix_gamma = False)
    relu = mx.symbol.Activation(name = name + '_relu', data = conv, act_type = 'relu')
    return relu

def residual(inputs, out_dim, stride, is_train, name):
    conv1 = conv(inputs, 3, out_dim, stride = stride, is_train = is_train, name = name+'_res1')
    conv2 = mx.symbol.Convolution(data = conv1, num_filter = out_dim, kernel = (3,3), pad = (1, 1), stride = (1, 1), no_bias = True, name = name + '_res2_conv')
    bn2 = mx.symbol.BatchNorm(name = name +'_res2_bn', data = conv2, use_global_stats = not is_train, fix_gamma = False)
    relu2 = mx.symbol.Activation(name = name + '_res2_relu', data = bn2, act_type = 'relu') 

    skip = mx.symbol.Convolution(name= name + '_res_skip_conv', data=inputs, num_filter=out_dim, pad=(0, 0), kernel=(1, 1), stride=(stride, stride), no_bias=True)
    skip = mx.symbol.BatchNorm(name = name + '_res_skip_bn', data=skip, use_global_stats=not is_train, fix_gamma=False)
    res = mx.symbol.broadcast_add(name= name + '_res_add', *[relu2, skip])
    res_relu = mx.symbol.Activation(name = name + '_res_out', data = res, act_type = 'relu')

    return res_relu
    
def make_layer(inputs, out_dim, modules, stride, is_train, layer = residual, name = ''):
    layers = inputs
    for i in range(modules):
        if i == 0:
            layers = layer(layers, out_dim, stride = stride, is_train = is_train, name = name +'_'+str(i)) 
        
        else:
            layers = layer(layers, out_dim, stride = 1, is_train = is_train, name = name +'_'+str(i)) 
    return layers

def make_layer_rever(inputs, in_dim, out_dim, modules, stride, is_train, layer = residual, name = ''):
    layers = inputs
    for i in range(modules):
        if i == modules - 1:
            layers = layer(layers, out_dim, stride = 1, is_train = is_train, name = name +'_'+str(i)) 
        
        else:
            layers = layer(layers, in_dim, stride = 1, is_train = is_train, name = name +'_'+str(i)) 
    return layers


def pre(inputs, is_train):
    pre_conv = conv(inputs, 7, 128, 2, is_train, 'pre')
    pre_residual = residual(pre_conv,  256,2, is_train, 'pre')
    return pre_residual
def Hourglass(inputs, n, dims, modules, is_train):
    curr_mod = modules[0]
    next_mod = modules[1]
    
    curr_dim = dims[0]
    next_dim = dims[1]

    up1 = make_layer(inputs, curr_dim, curr_mod, 1, is_train, residual, name = 'up1'+'_'+str(n))
    low1 = make_layer(up1, next_dim, curr_mod, 2, is_train, residual, name = 'low1'+'_'+str(n))
    low2 = Hourglass(low1, n - 1, dims = dims[1:], modules = modules[1:], is_train = is_train) if n > 1 else \
        make_layer(low1, next_dim, next_mod, 1, is_train, residual, name = 'low2'+'_'+str(n))
    low3 = make_layer_rever(low2, next_dim, curr_dim, curr_mod, 1, is_train, residual, name = 'low3'+ '_'+str(n))
    up2 = mx.symbol.UpSampling(low3, scale = 2, sample_type = 'nearest', name = 'up2'+ '_'+str(n))
    
    return up1 + up2
    

def corner_cnv(cnv, name, is_train):

    
    conv1 = conv(cnv, 3, 128, 1, is_train, name = name[0])
    conv2 = conv(cnv, 3, 128, 1, is_train, name = name[1])
    conv_skip = mx.symbol.Convolution(data = cnv, num_filter = 256, kernel = (1,1), pad = (0, 0), stride = (1, 1), no_bias = True, name = name + '_skip' + '_conv')
    bn_skip = mx.symbol.BatchNorm(name = name +'_skip' + '_bn', data = conv_skip, use_global_stats = not is_train, fix_gamma = False)
    pool1 = mx.sym.Custom(corner_input = conv1, corner_type = name[0], op_type = 'corner_pooling', name = 'corner_pooling_' + name[0])
    pool2 = mx.sym.Custom(corner_input = conv2, corner_type = name[1], op_type = 'corner_pooling', name = 'corner_pooling_' + name[1])
    pool_out = pool1 + pool2
    conv_p = mx.symbol.Convolution(data = pool_out, num_filter = 256, kernel = (3,3), pad = (1, 1), stride = (1, 1), no_bias = True, name = name + '_p' + '_conv')
    bn_p = mx.symbol.BatchNorm(name = name +'_p' + '_bn', data = conv_p, use_global_stats = not is_train, fix_gamma = False)
    pool_out
    relu = mx.symbol.Activation(name = name + '_relu', data = bn_skip + bn_p, act_type = 'relu')
    corner = conv(relu, 3,256, 1, is_train, name = name+'_out')
    return corner
    
def transpose_and_gather_feature(feat, ind, cfgs):
    feat = feat.transpose(0,2,3,1)
    feat = mx.sym.reshape(data = feat, shape = (0,-3,0))
    feat = mx.sym.gather_nd(feat, ind)
    feat = feat.reshape(data = feat, shape = (cfgs['batch_size'],cfgs['max_tag_len'], -1)
    return feat

def CornerNet(is_train, cfgs):
    data = mx.sym.Variable('data')
    tl_inds = mx.sym.Variable('tl_ind')
    br_inds = mx.sym.Variable('br_ind')
    inter = pre(data, is_train)
    
    dims = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    n = 5
    feat = Hourglass(inter, n , dims, modules, is_train)
    
    cnv = conv(inputs = feat, k = 3, out_dim = 256, stride = 1, is_train = is_train, name = 'cnv')
    tl_cnv = corner_cnv(cnv, name = 'tl', is_train = is_train)
    br_cnv = corner_cnv(cnv, name = 'br', is_train = is_train)

    tl_heat1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_heat1', with_bn = False)
    tl_heat_out = mx.symbol.Convolution(data = tl_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_heat_out')

    br_heat1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_heat1', with_bn = False)
    br_heat_out = mx.symbol.Convolution(data = br_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_heat_out')

    tl_tag1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_tag1', with_bn = False)
    tl_tag_out = mx.symbol.Convolution(data = tl_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_tag_out')

    br_tag1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_tag1', with_bn = False)
    br_tag_out = mx.symbol.Convolution(data = br_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_tag_out')

    tl_regrs1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_regrs1', with_bn = False)
    tl_regrs_out = mx.symbol.Convolution(data = tl_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_regrs_out')

    br_regrs1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_regrs1', with_bn = False)
    br_regrs_out = mx.symbol.Convolution(data = br_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_regrs_out')


    tl_tag = transpose_and_gather_feature(tl_tag_out, tl_inds, cfgs)
    br_tag = transpose_and_gather_feature(br_tag_out, br_inds, cfgs)
    tl_regr = transpose_and_gather_feature(tl_regr_out, tl_inds, cfgs)
    br_regr = transpose_and_gather_feature(br_regr_out, br_inds, cfgs)
    return mx.sym.Group([tl_cnv,br_cnv])


if __name__ == '__main__':
    if DEBUG:
        out = CornerNet(True, None)
        out = mx.sym.MakeLoss(out[0])
        mod = mx.mod.Module(symbol = out, context = mx.gpu(3), data_names=['data'],label_names=None)
        
        mod.bind(data_shapes=[('data',(32,3,256,256))])
# initialize parameters by uniform random numbers
        mod.init_params(initializer=mx.init.Uniform(scale=.1))
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
        metric = mx.metric.create('acc')
        inputs = np.random.randn(32,3,256,256)
        mod.forward(mx.io.DataBatch(data=[mx.nd.array(inputs)]),is_train = True)


        net_visualization(network='', num_classes = 80, train=True, output_dir = './', print_net = True, net = out)
