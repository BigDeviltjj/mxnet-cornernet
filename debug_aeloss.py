from symbols.cornernet import regr_loss, focal_loss, ae_loss, regr_loss,transpose_and_gather_feature
import mxnet as mx
import numpy as np
import random
cfg = {}
cfg['batch_size'] = 3
cfg['num_ctx'] = 1
cfg['max_tag_len'] = 128
np.random.seed(5)

pred = np.random.uniform(size = (3,2,128,128))
ind  =  np.zeros((3,128))
for i in range(3):
    ind[i,:i*30] = 1
p = mx.sym.Variable('p')
i = mx.sym.Variable('i')
out = transpose_and_gather_feature(p,i,cfg)
o = out.eval(ctx = mx.cpu(), p = mx.nd.array(pred), i = mx.nd.array(ind))
import pdb
pdb.set_trace()


out = [[] for _ in range(6)]
np.random.seed(5)
out[0] = np.random.uniform(size = (1,80,128,128))
out[1] = np.random.uniform(size = (1,80,128,128))
out[2] = np.random.uniform(size = (1,128,1))
out[3] = np.random.uniform(size = (1,128,1))
out[4] = np.random.uniform(size = (1,128,2))
out[5] = np.random.uniform(size = (1,128,2))



gt = [[] for _ in range(5)]
gt[0] = np.random.uniform(size = (1,80,128,128))
gt[1] = np.random.uniform(size = (1,80,128,128))
gt[2] = np.zeros((1,128))
gt[3] = np.random.uniform(size = (1,128,2))
gt[4] = np.random.uniform(size = (1,128,2))
a = [50,70,100]
for i in range(1):
    gt[2][i,:a[i]] = 1
gt[0][0,5,120,:] = 1

gt[0][0,5,:,120] = 1


out[0] = 1./(1 + np.exp(-out[0]))
out[1] = 1./(1 + np.exp(-out[1]))

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
e = mx.sym.Variable('e')
c = regr_loss(a,b,e)
print(c.eval(ctx = mx.cpu(),a = mx.nd.array(out[4]),b = mx.nd.array(gt[3]), e = mx.nd.array(gt[2])))
#print(d.eval(ctx = mx.cpu(),a = mx.nd.array(out[4]),b = mx.nd.array(out[5]), e = mx.nd.array(gt[2])))
