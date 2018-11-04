from symbols.cornernet import regr_loss
import mxnet as mx
import numpy as np
import random

pred = mx.sym.Variable('pred')
gt = mx.sym.Variable('gt')
m = mx.sym.Variable('m')
out =regr_loss(pred,gt,m)
p=mx.nd.array([[[1,2],[2,4],[3,6]],[[1,4],[2,1],[3,2]]])
g=mx.nd.array([[[3,1],[0,4],[4,5]],[[1.5,3.4],[2,1],[0,2]]])
mask=mx.nd.array([[1,1,1],[1,1,0]])
print(p,g,mask)
print(out.eval(ctx = mx.cpu(), pred = p, gt = g,m=mask))
#pred = mx.sym.Variable('pred')
#gt = mx.sym.Variable('gt')
#p = mx.nd.arange(1,26).reshape((1,1,5,5))/25
#g = mx.nd.arange(1,26)[::-1].reshape((1,1,5,5))/25
#g[0,0,0] = 1.
#l = focal_loss(pred,gt)
#print(p,g)
#print(l.eval(ctx = mx.cpu(), pred = p, gt = g))
#tag0= mx.sym.Variable('tag0')
#tag1= mx.sym.Variable('tag1')
#mask = mx.sym.Variable('mask')
#pull,push = ae_loss(tag0, tag1, mask)
#n = 2
#d = 5
#tag0 = np.random.randn(n,d,1)
#tag1 = np.random.randn(n,d,1)
#print(tag0,tag1)
#tag0 = mx.nd.array(tag0)
#tag1 = mx.nd.array(tag1)
#mask = mx.nd.zeros((n,d))
#for i in range(n):
#    ones = random.randint(1,d-1)
#    mask[i][:ones] = 1
#print(tag0, tag1, mask)
#out = mx.sym.Group([pull, push])

#print(out.eval(ctx = mx.cpu(), tag0 = tag0, tag1 = tag1, mask = mask))
#exe = out.simple_bind(ctx =mx.cpu(), tag0 = tag0, tag1 = tag1, mask = mask)
#exe.forward(a=a,b=b)
#print(exe.outputs)
#mod = mx.mod.Module(b,data_names=['a'])
#mod.bind(data_shapes=[('a',(1,1,5,5))], force_rebind = True)
#mod.init_params()
#mod.forward( mx.io.DataBatch(data=[a]),is_train=False )
#print(mod.get_outputs())


