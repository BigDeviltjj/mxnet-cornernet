import numpy as np
import mxnet as mx

def _nms(heat, kernel = 3):
    pad = (kernel - 1)//2
    hmax = mx.nd.Pooling(data = heat, kernel = (kernel,kernel),stride = (1,1), pad = (pad,pad))
    ret = heat == hmax
    ret = ret * heat
    return ret
def _topk(scores, K = 20):
    b, c, h, w = scores.shape
    score_t = scores.reshape((b,-1))
    topk_inds = score_t.argsort(axis = 1)[:,-K:][:,::-1]
    score_t.sort(axis = 1)
    topk_scores = score_t[:,-K:][:,::-1]
    topk_clses = (topk_inds/(w*h)).astype(int)
    topk_inds = topk_inds % (w*h)
    topk_ys = (topk_inds/w).astype(int)
    topk_xs = (topk_inds%w).astype(int)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def gather_numpy(x, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = x.shape[:dim] + x.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(x, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def _gather_feat(feat, ind):
    b, l = ind.shape
    ind0 = np.repeat(np.arange(b),l)
    ind1 = ind.reshape((-1))
    index = np.stack([ind0,ind1])
    print(feat.shape)
    feat= feat[ind0,ind1,:]
    feat = feat.reshape(b,l,-1)
#    ind = np.tile(ind[:,:,None],(1,1,dim))
    return feat
def _transpose_and_gather_feat(feat, ind):
    feat = feat.transpose((0,2,3,1))
    feat = feat.reshape((feat.shape[0],-1,feat.shape[-1]))
    feat = _gather_feat(feat, ind)
    return feat

def decode(tl_heat, br_heat,
           tl_tag, br_tag,
           tl_regr, br_regr,
           K = 100, kernel = 3,
           ae_threshold = 0.5,
           num_dets = 1000):

    import pdb
    pdb.set_trace()
    tl_heat = mx.nd.sigmoid(tl_heat)
    br_heat = mx.nd.sigmoid(br_heat)

    tl_heat = _nms(tl_heat, kernel)
    br_heat = _nms(br_heat, kernel)

    tl_heat = tl_heat.asnumpy()
    br_heat = br_heat.asnumpy()
    tl_tag = tl_tag.asnumpy()
    br_tag =  br_tag.asnumpy()
    tl_regr = tl_regr.asnumpy()
    br_regr =  br_regr.asnumpy()
    b, c, h, w = tl_heat.shape
    
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    tl_ys = np.tile(tl_ys[:,:,None],(1,1,K))
    tl_xs = np.tile(tl_xs[:,:,None],(1,1,K))
    br_ys = np.tile(br_ys[:,None,:],(1,K,1))
    br_xs = np.tile(br_xs[:,None,:],(1,K,1))

    tl_regr = _transpose_and_gather_feat(tl_regr, tl_inds)
    br_regr = _transpose_and_gather_feat(br_regr, br_inds)
    tl_regr = tl_regr.reshape((b,K,1,2))
    br_regr = br_regr.reshape((b,1,K,2))

    tl_xs = tl_xs + tl_regr[:,:,:,0]
    tl_ys = tl_ys + tl_regr[:,:,:,1]
    br_xs = br_xs + br_regr[:,:,:,0]
    br_ys = br_ys + br_regr[:,:,:,1]

    bboxes = np.stack([tl_xs, tl_ys, br_xs, br_ys],axis = 3)  #b,k,k,4
    tl_tag = _transpose_and_gather_feat(tl_tag, tl_inds).reshape(b,K,1)
    br_tag = _transpose_and_gather_feat(br_tag, br_inds).reshape(b,1,K)
    dists = np.abs(tl_tag - br_tag)


    tl_scores = np.tile(tl_scores.reshape((b,K,1)),(1,1,K))
    br_scores = np.tile(br_scores.reshape((b,1,K)),(1,K,1))
    scores = (tl_scores + br_scores)/2


    cls_inds = tl_clses.reshape((b,K,1)) != br_clses.reshape((b,1,K))

    dist_inds = (dists > ae_threshold)

    width_inds = br_xs < tl_xs
    height_inds = br_ys < tl_ys

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1
    scores = scores.reshape(b,-1)


    import pdb
    pdb.set_trace()
    inds = scores.argsort(axis = 1)[:,-num_dets:][:,::-1]
    scores.sort(axis = 1)
    scores = scores[:,-num_dets:][:,::-1]
    bboxes = bboxes.reshape((b,-1,4))
    bboxes = _gather_feat(bboxes, inds)

    clses = np.tile(tl_clses.reshape((b,K,1)),(1,1,K)).reshape((b,-1,1))
    clses = _gather_feat(clses, inds)

    tl_scores = tl_scores.reshape((b, -1, 1))
    tl_scores = _gather_feat(tl_scores, inds)
    br_scores = br_scores.reshape((b, -1, 1))
    br_scores = _gather_feat(br_scores, inds)

    print(bboxes.shape)
    print(scores.shape)
    print(tl_scores.shape)
    print(clses.shape)
    detections = np.concatenate([bboxes, scores[:,:,None], tl_scores, br_scores, clses], axis = 2)
    return detections
