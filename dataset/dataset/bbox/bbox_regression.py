import numpy as np

def expand_bbox_regression_targets(bbox_targets_data, num_classes,cfg):
    ret = np.zeros((bbox_targets_data.shape[0],num_classes*4),dtype = np.float32)
    bbox_weights = np.zeros(ret.shape,dtype=np.float32)
    for i,k in enumerate(bbox_targets_data):
        index = int(k[0])

        if index is not 0:
            ret[i][(index*4):(index*4+4)] = k[1:]
            bbox_weights[i][(index*4):(index*4+4)] = cfg.TRAIN.BBOX_WEIGHTS
    return ret,bbox_weights
