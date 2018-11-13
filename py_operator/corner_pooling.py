import mxnet as mx
import numpy as np
DEBUG = True

class CornerPoolingOperator(mx.operator.CustomOp):
    def __init__(self, name):
        super(CornerPoolingOperator,self).__init__()
        self._type = name
    def forward(self, is_train, req, in_data, out_data, aux):

        b, c, h, w = in_data[0].shape
        inputs = in_data[0]
        self.in_grad = mx.nd.zeros_like(inputs)
        mx.autograd.mark_variables(inputs, self.in_grad)
        self.outputs = []
        if self._type == 'l' or self._type == 'r':
            max_val = mx.nd.full(shape = inputs[:,:,:,0].shape,val = float('-inf'))
            max_val = mx.nd.expand_dims(max_val, axis = 3)
        elif self._type == 't' or self._type == 'b':
            max_val = mx.nd.full(shape = inputs[:,:,0,:].shape,val = float('-inf'))
            max_val = mx.nd.expand_dims(max_val, axis = 2)
        with mx.autograd.record():
            
            if self._type == 'l' or self._type == 'r':
    
                if self._type == 'l':
                    s = w - 1
                    e = -1
                    stride = -1
                if self._type == 'r':
                    s = 0
                    e = w
                    stride = 1
                for i in range(s,e,stride):
                    #cur_val = mx.nd.take(inputs,indices = mx.nd.array([i]),axis = 3)
                    cur_val = mx.nd.slice(inputs,begin = (0,0,0,i), end = (None,None,None, i + 1))
                    max_val = mx.nd.maximum(max_val, cur_val) 
                    if self._type == 'r' :
                        self.outputs.append(max_val)
                    else: self.outputs.insert(0, max_val)
           
            elif self._type == 't' or self._type == 'b':
    
                if self._type == 't':
                    s = h - 1
                    e = -1
                    stride = -1
                if self._type == 'b':
                    s = 0
                    e = h
                    stride = 1
                for i in range(s,e,stride):

                    cur_val = mx.nd.slice(inputs,begin = (0,0,i,0), end = (None,None, i+1, None))
                    max_val = mx.nd.maximum(max_val, cur_val) 
                    if self._type == 'b':
                        self.outputs.append(max_val)
                    else: self.outputs.insert(0, max_val)
            concat_axis = 3 if self._type in 'lr' else 2
            self.output = mx.nd.concat(*self.outputs, dim = concat_axis)

        self.assign(out_data[0], req[0], self.output)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        output_grad = out_grad[0]
        b, c, h, w = out_grad[0].shape
        with mx.autograd.record():
            flow = out_grad[0] * self.output
            flow.backward()
            #(self.output).backward()
        
        self.assign(in_grad[0], req[0], self.in_grad)

@mx.operator.register('corner_pooling')
class CornerPoolingProp(mx.operator.CustomOpProp):
    def __init__(self, corner_type):
        super(CornerPoolingProp,self).__init__()
        self._type = corner_type
    def list_arguments(self):
        return ['corner_input']
    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, in_shape
    def create_operator(self, ctx, shapes, dtypes):
        return CornerPoolingOperator(self._type)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return []


if __name__ == '__main__':
    if DEBUG:
        mx.random.seed(123)
        a = mx.sym.Variable('a')
        b = mx.sym.Custom(corner_input = a, corner_type = 'b', op_type = 'corner_pooling', name = 'corner')
        b = mx.sym.MakeLoss(b)
        exe = b.simple_bind(ctx = mx.cpu(),a = (2,2,5,5))
        it = mx.nd.random.shuffle(mx.nd.arange(100)).reshape((2,2,5,5))
        print(it)

        exe.forward(is_train = True, a=it)
        exe.backward()
        print('output',exe.outputs)
        print('grad', exe.grad_arrays)
        
        import pdb
        pdb.set_trace()
        print(b.infer_shape(a=(1,2,3,4,5)))
        print(b.list_arguments())
        print(b.list_outputs())
        c = 1
