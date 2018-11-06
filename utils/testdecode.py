import numpy as np
import decode
import mxnet as mx
np.random.seed(3)
tl_heat = mx.nd.array(np.random.randn(3,80,128,128))
br_heat = mx.nd.array(np.random.randn(3,80,128,128))
tl_tag = mx.nd.array(np.random.randn(3,1,128,128))
br_tag = mx.nd.array(np.random.randn(3,1,128,128))
tl_regr = mx.nd.array(np.random.randn(3,2,128,128))
br_regr = mx.nd.array(np.random.randn(3,2,128,128))

import pdb
pdb.set_trace()
K = 100
kernel = 3
ae_threshold = 0.5
num_dets = 1000
detection = decode.decode(tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,K, kernel,ae_threshold, num_dets)
print(detection)
