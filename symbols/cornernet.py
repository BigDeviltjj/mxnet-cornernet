from __future__ import division
import mxnet as mx
import numpy as np
import sys
import os
sys.path.insert(0,os.path.join(os.path.abspath('./'),'./'))
#print(__file__)
#print(sys.path)
from py_operator.corner_pooling import *
from config.cfg import cfg

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

def residual(inputs, out_dim, stride, is_train, name, revr = False):
    conv1 = conv(inputs, 3, out_dim, stride = stride, is_train = is_train, name = name+'_res1')
    conv2 = mx.symbol.Convolution(data = conv1, num_filter = out_dim, kernel = (3,3), pad = (1, 1), stride = (1, 1), no_bias = True, name = name + '_res2_conv')
    bn2 = mx.symbol.BatchNorm(name = name +'_res2_bn', data = conv2, use_global_stats = not is_train, fix_gamma = False)
    relu2 = mx.symbol.Activation(name = name + '_res2_relu', data = bn2, act_type = 'relu') 

    if stride == 1 and not revr:
        skip = inputs
    else:
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
    revr = not (in_dim == out_dim)
    for i in range(modules):
        if i == modules - 1:
            layers = layer(layers, out_dim, stride = 1, is_train = is_train, name = name +'_'+str(i), revr = revr) 
        
        else:
            layers = layer(layers, in_dim, stride = 1, is_train = is_train, name = name +'_'+str(i), revr = False) 
    return layers


def pre(inputs, is_train):
    pre_conv = conv(inputs, 7, 128, 2, is_train, 'pre')
    pre_residual = residual(pre_conv,  256,2, is_train, 'pre')
    return pre_residual
def Hourglass(inputs, n, dims, modules, is_train, stage = 0):
    curr_mod = modules[0]
    next_mod = modules[1]
    
    curr_dim = dims[0]
    next_dim = dims[1]

    up1 = make_layer(inputs, curr_dim, curr_mod, 1, is_train, residual, name = 'up_'+str(stage)+'_1'+'_'+str(n))
    low1 = make_layer(up1, next_dim, curr_mod, 2, is_train, residual, name = 'low_'+str(stage)+'_1'+'_'+str(n))
    low2 = Hourglass(low1, n - 1, dims = dims[1:], modules = modules[1:], is_train = is_train,stage = stage) if n > 1 else \
        make_layer(low1, next_dim, next_mod, 1, is_train, residual, name = 'low_'+str(stage)+'_2'+'_'+str(n))
    low3 = make_layer_rever(low2, next_dim, curr_dim, curr_mod, 1, is_train, residual, name = 'low_'+str(stage)+'_3'+ '_'+str(n))
    up2 = mx.symbol.UpSampling(low3, scale = 2, sample_type = 'nearest', name = 'up_'+str(stage)+'_2'+ '_'+str(n))
    
    return up1 + up2
    

def corner_cnv(cnv, name, is_train):

    
    conv1 = conv(cnv, 3, 128, 1, is_train, name = name + '_p1_conv1')
    conv2 = conv(cnv, 3, 128, 1, is_train, name = name + '_p2_conv1')
    conv_skip = mx.symbol.Convolution(data = cnv, num_filter = 256, kernel = (1,1), pad = (0, 0), stride = (1, 1), no_bias = True, name = name + '_conv1')
    bn_skip = mx.symbol.BatchNorm(name = name + '_bn1', data = conv_skip, use_global_stats = not is_train, fix_gamma = False)
    pool1 = mx.sym.Custom(corner_input = conv1, corner_type = name[0], op_type = 'corner_pooling', name = 'corner_pooling_' + name[0]+name[-2:])
    pool2 = mx.sym.Custom(corner_input = conv2, corner_type = name[1], op_type = 'corner_pooling', name = 'corner_pooling_' + name[1]+name[-2:])
    pool_out = pool1 + pool2
    conv_p = mx.symbol.Convolution(data = pool_out, num_filter = 256, kernel = (3,3), pad = (1, 1), stride = (1, 1), no_bias = True, name = name + '_p' + '_conv1')
    bn_p = mx.symbol.BatchNorm(name = name +'_p' + '_bn1', data = conv_p, use_global_stats = not is_train, fix_gamma = False)
    relu = mx.symbol.Activation(name = name + '_relu', data = bn_skip + bn_p, act_type = 'relu')
    corner = conv(relu, 3,256, 1, is_train, name = name+'_conv2')
    return corner
    
def transpose_and_gather_feature(feat, ind, cfgs):
    feat = feat.transpose((0,2,3,1))
    feat = mx.sym.reshape(data = feat, shape = (0,-3,0))
    ind_1 = ind.reshape(-1)
    ind_0 = mx.sym.arange(cfgs['batch_size']//cfgs['num_ctx'])
    ind_0 = mx.sym.transpose(mx.sym.tile(ind_0, (cfgs['max_tag_len'], 1))).reshape(-1)
    index = mx.sym.stack(ind_0, ind_1)
    feat = mx.sym.gather_nd(feat, index)
    feat = feat.reshape((cfgs['batch_size']//cfgs['num_ctx'],cfgs['max_tag_len'], -1))
    return feat

def focal_loss(pred, gt):
    eps = 1e-5
    neg_weights = mx.sym.pow(1 - gt, 4)
    #neg_weights = mx.sym.broadcast_power(1 - gt + eps, mx.sym.full(shape=(1,),val=4))

    #loss = 0
    zero = mx.sym.zeros_like(gt)
    pos_loss = mx.sym.where(gt == 1., mx.sym.log(pred + eps) * mx.sym.pow(1-pred, 2), zero)
    neg_loss = mx.sym.where(gt < 1. , mx.sym.log(1 - pred + eps) * mx.sym.pow(pred+eps, 2) * neg_weights, zero)

    num_pos = (gt == 1.).sum()
    loss = mx.sym.where(num_pos == 0., -neg_loss.sum(), -(pos_loss.sum() + neg_loss.sum()) / num_pos)
    return loss

def ae_loss(tag0, tag1, mask):
    num = mx.sym.sum(mask, axis = 1,keepdims = True)
    tag0 = tag0.squeeze(axis = -1)
    tag1 = tag1.squeeze(axis = -1)

    tag_mean = (tag0 + tag1) / 2

    tag0 = mx.sym.broadcast_div(mx.sym.pow(tag0 - tag_mean, 2) , (num + 1e-4))
    tag0 = mx.sym.broadcast_mul(tag0, mask)
    tag0 = tag0.sum()
    tag1 = mx.sym.broadcast_div(mx.sym.pow(tag1 - tag_mean, 2) , (num + 1e-4))
    tag1 = mx.sym.broadcast_mul(tag1, mask)
    tag1 = tag1.sum()
    pull = tag0 + tag1

    push_mask = mx.sym.broadcast_add(mx.sym.expand_dims(mask, axis = 1), mx.sym.expand_dims(mask, axis = 2))

    push_mask = push_mask == 2
    num = mx.sym.expand_dims(num, axis = 2)
    num2 = (num - 1) * num
    dist = mx.sym.broadcast_sub(mx.sym.expand_dims(tag_mean, 1) , mx.sym.expand_dims(tag_mean,2))
    dist = 1 - mx.sym.abs(dist)
    dist = mx.sym.Activation(data = dist, act_type = 'relu')
    dist = mx.sym.broadcast_sub(dist, 1/(num + 1e-4))
    dist = mx.sym.broadcast_div(dist, num2 + 1e-4)
    dist = mx.sym.broadcast_mul(dist, push_mask)
    push = dist.sum()
    return pull, push

def regr_loss(regr, gt_regr, mask):
    num = mask.sum()
    mask = mask.expand_dims(axis = 2).broadcast_like(gt_regr)
    loss = mask * mx.sym.smooth_l1(data=(gt_regr - regr), scalar = 1.)
    loss = loss.sum() / (num + 1e-4)
    return loss

def AELoss(tl_heat_pred, br_heat_pred,
           tl_tag_pred, br_tag_pred,
	   tl_regr_pred, br_regr_pred,
	   tl_heat_gt, br_heat_gt,
	   tl_regr_gt, br_regr_gt,
           masks, cfgs):
    tl_heat_pred = mx.symbol.sigmoid(tl_heat_pred)
    br_heat_pred = mx.symbol.sigmoid(br_heat_pred)

    
    tl_heat_loss = focal_loss(tl_heat_pred, tl_heat_gt)
    br_heat_loss = focal_loss(br_heat_pred, br_heat_gt)

    pull_loss, push_loss = ae_loss(tl_tag_pred, br_tag_pred, masks)
    pull_loss = pull_loss * cfgs['pull_weight']
    push_loss = push_loss * cfgs['push_weight']

    regr_l = regr_loss(tl_regr_pred, tl_regr_gt, masks) + regr_loss(br_regr_pred, br_regr_gt, masks)
    regr_l = regr_l * cfgs['regr_weight']
    loss = tl_heat_loss + br_heat_loss + pull_loss + push_loss + regr_l
    return loss



def CornerNet(is_train, cfgs):
    data = mx.sym.Variable('data')
    tl_inds = mx.sym.Variable('tl_inds')
    br_inds = mx.sym.Variable('br_inds')
    tl_heatmaps = mx.sym.Variable('tl_heatmaps')
    br_heatmaps = mx.sym.Variable('br_heatmaps')
    tl_regrs = mx.sym.Variable('tl_regrs')
    br_regrs = mx.sym.Variable('br_regrs')
    tag_masks = mx.sym.Variable('tag_masks')

    inter = pre(data, is_train)
    
    dims = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    n = 5
#stage 0
    feat = Hourglass(inter, n , dims, modules, is_train)
    
    cnv = conv(inputs = feat, k = 3, out_dim = 256, stride = 1, is_train = is_train, name = 'cnv_0')
    tl_cnv = corner_cnv(cnv, name = 'tl_0', is_train = is_train)
    br_cnv = corner_cnv(cnv, name = 'br_0', is_train = is_train)

    tl_heat1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_heats_0_0', with_bn = False)
    tl_heat_out = mx.symbol.Convolution(data = tl_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_heats_0_1')

    br_heat1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_heats_0_0', with_bn = False)
    br_heat_out = mx.symbol.Convolution(data = br_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_heats_0_1')

    tl_tag1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_tags_0_0', with_bn = False)
    tl_tag_out = mx.symbol.Convolution(data = tl_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_tags_0_1')

    br_tag1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_tags_0_0', with_bn = False)
    br_tag_out = mx.symbol.Convolution(data = br_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_tags_0_1')

    tl_regrs1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_regrs_0_0', with_bn = False)
    tl_regrs_out = mx.symbol.Convolution(data = tl_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_regrs_0_1')

    br_regrs1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_regrs_0_0', with_bn = False)
    br_regrs_out = mx.symbol.Convolution(data = br_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_regrs_0_1')


    tl_tag = transpose_and_gather_feature(tl_tag_out, tl_inds, cfgs)
    br_tag = transpose_and_gather_feature(br_tag_out, br_inds, cfgs)
    tl_regr = transpose_and_gather_feature(tl_regrs_out, tl_inds, cfgs)
    br_regr = transpose_and_gather_feature(br_regrs_out, br_inds, cfgs)

    loss0 = AELoss(tl_heat_out, br_heat_out, tl_tag, br_tag, tl_regr, br_regr,tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tag_masks, cfgs)
#inter stage
    inter = mx.symbol.Convolution(data = inter, num_filter = 256, kernel = (1,1), pad = (0, 0), stride = (1, 1), no_bias = True, name = 'inter_0_0')
    inter = mx.symbol.BatchNorm(name = 'inter_0_1', data = inter, use_global_stats = not is_train, fix_gamma = False)
    cnv0 = mx.symbol.Convolution(data = cnv, num_filter = 256, kernel = (1,1), pad = (0, 0), stride = (1, 1), no_bias = True, name = 'cnvs_0_0')
    cnv0 = mx.symbol.BatchNorm(name = 'cnvs_0_1', data = cnv0, use_global_stats = not is_train, fix_gamma = False)
    inter = inter + cnv0
    inter = mx.symbol.Activation(name = 'inter_0_relu', data = inter, act_type = 'relu')
    inter = residual(inter, 256, 1, is_train, 'inters_0')
#stage 1

    feat = Hourglass(inter, n , dims, modules, is_train,stage = 1)
    
    cnv = conv(inputs = feat, k = 3, out_dim = 256, stride = 1, is_train = is_train, name = 'cnv_1')
    tl_cnv = corner_cnv(cnv, name = 'tl_1', is_train = is_train)
    br_cnv = corner_cnv(cnv, name = 'br_1', is_train = is_train)

    tl_heat1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_heats_1_0', with_bn = False)
    tl_heat_out = mx.symbol.Convolution(data = tl_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_heats_1_1')

    br_heat1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_heats_1_0', with_bn = False)
    br_heat_out = mx.symbol.Convolution(data = br_heat1, kernel = (1, 1), num_filter = cfgs['num_classes'], pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_heats_1_1')

    tl_tag1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_tags_1_0', with_bn = False)
    tl_tag_out = mx.symbol.Convolution(data = tl_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_tags_1_1')

    br_tag1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_tags_1_0', with_bn = False)
    br_tag_out = mx.symbol.Convolution(data = br_tag1, kernel = (1, 1), num_filter = 1, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_tags_1_1')

    tl_regrs1 = conv(tl_cnv, 3, 256, 1, is_train,name = 'tl_regrs_1_0', with_bn = False)
    tl_regrs_out = mx.symbol.Convolution(data = tl_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'tl_regrs_1_1')

    br_regrs1 = conv(br_cnv, 3, 256, 1, is_train,name = 'br_regrs_1_0', with_bn = False)
    br_regrs_out = mx.symbol.Convolution(data = br_regrs1, kernel = (1, 1), num_filter = 2, pad = (0, 0), stride = (1, 1), no_bias = False, name = 'br_regrs_1_1')

    tl_tag = transpose_and_gather_feature(tl_tag_out, tl_inds, cfgs)
    br_tag = transpose_and_gather_feature(br_tag_out, br_inds, cfgs)
    tl_regr = transpose_and_gather_feature(tl_regrs_out, tl_inds, cfgs)
    br_regr = transpose_and_gather_feature(br_regrs_out, br_inds, cfgs)

    loss1 = AELoss(tl_heat_out, br_heat_out, tl_tag, br_tag, tl_regr, br_regr,tl_heatmaps, br_heatmaps, tl_regrs, br_regrs, tag_masks, cfgs)
    loss = mx.sym.MakeLoss(loss0 + loss1)
#####remove##########
#    tag_masks = mx.sym.expand_dims(tag_masks,axis = 2)
#    tl_heatmap_loss = mx.sym.smooth_l1(tl_heatmaps - tl_heat_out,scalar = 1)
#    br_heatmap_loss = mx.sym.smooth_l1(br_heatmaps - br_heat_out,scalar = 1)
#    tl_reg_loss = mx.sym.broadcast_mul(tag_masks , mx.sym.smooth_l1(tl_regrs - tl_regr,scalar = 1))
#    br_reg_loss = mx.sym.broadcast_mul(tag_masks , mx.sym.smooth_l1(br_regrs - br_regr,scalar = 1))
#    tl_tag_loss = mx.sym.broadcast_mul(tag_masks , mx.sym.smooth_l1(tl_inds - tl_tag,scalar = 1))
#    br_tag_loss = mx.sym.broadcast_mul(tag_masks , mx.sym.smooth_l1(br_inds - br_tag,scalar = 1))
#
#    tl_heatmap_loss = mx.symbol.MakeLoss(tl_heatmap_loss)
#    br_heatmap_loss = mx.symbol.MakeLoss(br_heatmap_loss)
#    tl_reg_loss = mx.symbol.MakeLoss(tl_reg_loss)
#    br_reg_loss = mx.symbol.MakeLoss(br_reg_loss)

    return mx.sym.Group([loss,mx.sym.BlockGrad(tl_heat_out), mx.sym.BlockGrad(br_heat_out), mx.sym.BlockGrad(tl_tag_out), mx.sym.BlockGrad(br_tag_out), mx.sym.BlockGrad(tl_regrs_out), mx.sym.BlockGrad(br_regrs_out)])#mx.sym.Group([tl_heatmap_loss, br_heatmap_loss, tl_reg_loss, br_reg_loss,tl_regr, tl_regrs_out])


if __name__ == '__main__':
    if DEBUG:
        cfg['network']['num_ctx'] = 2
        out = CornerNet(True, cfg['network'])
        net_visualization(network='', num_classes = 80, train=True, output_dir = './', print_net = True, net = out)
        mod = mx.mod.Module(symbol = out[0], context = [mx.gpu(0),mx.gpu(1)], data_names=['data'],label_names=['tl_heatmaps', 'br_heatmaps', 'tl_regrs', 'br_regrs', 'tl_inds', 'br_inds', 'tag_masks'])
        
        mod.bind(data_shapes=[('data',(4,3,511,511))],
                label_shapes = [('tl_heatmaps', (4, 80, 128, 128)), ('br_heatmaps', (4, 80, 128, 128)), ('tl_regrs', (4, 128, 2)), ('br_regrs', (4, 128, 2)), ('tl_inds', (4, 128)), ('br_inds', (4, 128)), ('tag_masks', (4, 128))])
# initialize parameters by uniform random numbers
        mod.init_params()
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
        a = mod.get_params()
        for i in a[0].keys():
            print(i)
        import pdb
        pdb.set_trace()
        metric = mx.metric.create('acc')
        inputs = np.random.randn(32,3,256,256)
        mod.forward(mx.io.DataBatch(data=[mx.nd.array(inputs)]),is_train = True)
        print(mod.get_outputs)


