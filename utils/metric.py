import mxnet as mx
import numpy as np

class CornerNetMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(CornerNetMetric, self).__init__('CornerNet')
        self.name = ['Loss']
        self.reset()
    
    def reset(self):
        self.num_inst = 0.
        self.sum_metric = 0.

    def update(self, labels, preds):
        loss = preds[0][0].asnumpy()
        self.sum_metric += loss
        self.num_inst += 1
    
    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

