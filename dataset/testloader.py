from __future__ import division
import numpy as np
import mxnet as mx
import cv2
from .utils import gaussian_radius, draw_gaussian
from .image import get_test_image


class TestDetRecordIter(mx.io.DataIter):
    def __init__(self,roidb, test_params,batch_size, shuffle
                 ):
        super(TestDetRecordIter, self).__init__()
        self.roidb = roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        self.test_params = test_params
        self.batch_size = batch_size
        assert batch_size == 1, 'testloader only support batch size = 1'
        self.shuffle = shuffle
        self.provide_label = None
        self.data_names = ['data']
        self.label_names = None
        self.label = None
        self.cur = 0
        self._get_batch()
        self.reset()

    @property
    def provide_data(self):
        return [(k,v.shape) for k,v in zip(self.data_names, self.data)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            info = self._get_batch()
            self.cur += self.batch_size
            return info, mx.io.DataBatch(data = self.data, label = self.label, 
                                   pad = self.getpad(),index = self.getindex(),
                                   provide_data = self.provide_data, provide_label = self.provide_label)
        else:
            raise StopIteration
    def getindex(self):
        return self.cur // self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0


    def _get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        ims, info = get_test_image(roidb,self.test_params)
        self.data =[mx.nd.array(np.concatenate(ims,axis = 0))]
        return info
            
        
