from __future__ import division
import numpy as np
import mxnet as mx
import cv2
from .utils import gaussian_radius, draw_gaussian
from .image import get_image
import multiprocessing as mp
#mp.set_start_method('spawn',force = True)
import threading
import time


class DetRecordIter(mx.io.DataIter):
    def __init__(self,roidb, train_params,batch_size, shuffle, num_workers = 4, pin_memory = False):
        super(DetRecordIter, self).__init__()
        self.roidb = roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        self.train_params = train_params
        self.batch_size = batch_size
        self.data_shape =(3,) + train_params['input_size']
        self.shuffle = shuffle
        self.provide_label = None
        self.data_names = ['data']
        self.label_names = ['tl_heatmaps','br_heatmaps','tl_regrs','br_regrs','tl_inds','br_inds', 'tag_masks']
        self._num_workers = num_workers
        self.pin_memory = pin_memory
        self.reset()
        self.next()


    def _push_next(self):
        s = self.cur
        e = self.cur + self.batch_size
        r = (s,e)
        if e > self.size:
            return
        self.cur += self.batch_size
        self._key_queue.put((self._sent_idx, r))
        self._sent_idx += 1

#        self._get_batch()
#        if not self.provide_label:
#            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)

    def worker_loop(self, key_queue, data_queue):
        while True:
            idx, dur = key_queue.get()
            if idx is None:
                break
            s,e = dur
            t = time.time()
            batch = self._get_batch(s,e)
            data_queue.put((idx, batch))
    def fetcher_loop(self, data_queue, data_buffer, pin_memory = False):
        while True:
            idx, batch = data_queue.get()
            t = time.time()
            if idx is None:
                return
            if pin_memory:
                batch = ([d.as_in_context(mx.cpu_pinned()) for d in batch[0]], [d.as_in_context(mx.cpu_pinned()) for d in batch[1]])
#            else:
#                batch = ([d.as_in_context(mx.cpu()) for d in batch[0]], [d.as_in_context(mx.cpu()) for d in batch[1]])
            data_buffer[idx] = batch




    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.index)
        self.cur = 0
        self._key_queue = mp.Queue()
        self._data_queue = mp.Queue()
#        self._data_buffer = {}
        self._data_buffer = mp.Manager().dict()
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._shutdown = False
        workers = []
        for _ in range(self._num_workers):
            worker = mp.Process(target = self.worker_loop, args = (self._key_queue, self._data_queue))
            worker.daemon = True
            worker.start()
            workers.append(worker)

#        self._fetcher = threading.Thread(
        self._fetcher = mp.Process(
                target = self.fetcher_loop,
                args = (self._data_queue, self._data_buffer, self.pin_memory))
        self._fetcher.daemon = True
        self._fetcher.start()
        for _ in range(2 * self._num_workers):
            self._push_next()

#    def iter_next(self):
#        return self.cur + self.batch_size <= self.size

    def next(self):
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'data buffer shold be empty'
            self.shutdown()
            raise StopIteration

        while True:
            if self._rcvd_idx in self._data_buffer:
                batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx +=1
                self._push_next()
                self.provide_data = [(k,v.shape) for k,v in zip(self.data_names, batch[0])]
                self.provide_label = [(k,v.shape) for k,v in zip(self.label_names, batch[1])]
                return mx.io.DataBatch(data = batch[0], label = batch[1], 
                                   pad = self.getpad(),
                                   provide_data = self.provide_data, provide_label = self.provide_label)

    def getindex(self):
        return self.cur // self.batch_size


    def _get_batch(self, start, end):
        cur_from = start
        cur_to = end
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        ims, roidbs = get_image(roidb,self.train_params)
        data =[mx.nd.array(np.concatenate(ims,axis = 0))]
        

        train_params = self.train_params
        tl_heatmaps = np.zeros((self.batch_size, train_params['num_classes'],train_params['output_size'][0], train_params['output_size'][1]))
        br_heatmaps = np.zeros((self.batch_size, train_params['num_classes'],train_params['output_size'][0], train_params['output_size'][1]))
        tl_regrs = np.zeros((self.batch_size, train_params['max_tag_len'],2))
        br_regrs = np.zeros((self.batch_size, train_params['max_tag_len'],2))
        # tl_inds = np.zeros((2, self.batch_size * train_params['max_tag_len']))
        # br_inds = np.zeros((2, self.batch_size * train_params['max_tag_len']))
        tl_inds = np.zeros((self.batch_size, train_params['max_tag_len']))
        br_inds = np.zeros((self.batch_size, train_params['max_tag_len']))

        tag_masks = np.zeros((self.batch_size, train_params['max_tag_len']))
        tag_lens = np.zeros((self.batch_size,), dtype = np.int64)
        width_ratio = train_params['output_size'][0] / self.data_shape[2]
        height_ratio = train_params['output_size'][1] / self.data_shape[1]
        for b, single_label in enumerate(roidbs):
            keep = np.where(single_label[:,-1]>-1)[0]
            gt_boxes = single_label[keep]
            for gt_box in gt_boxes:

                box = gt_box[:4]
                feat_box = box * np.array([width_ratio, height_ratio, width_ratio, height_ratio])
                feat_box_quanti = feat_box.astype(int)
                
                reg_label = feat_box - feat_box_quanti
                cls = int(gt_box[-1])
                
                
                if train_params['gaussian_bump']:
                    width = np.ceil((gt_box[2] - gt_box[0])*width_ratio)
                    height = np.ceil(( gt_box[3] - gt_box[1])*height_ratio)
                    if train_params['gaussian_radius'] == -1:
                        radius = gaussian_radius((width, height), train_params['gaussian_iou'])
                        radius = max(0, int(radius))
                    else:
                        radius = train_params['gaussian_radius']
                    draw_gaussian(tl_heatmaps[b, cls] ,feat_box_quanti[:2], radius)
                    draw_gaussian(br_heatmaps[b, cls] ,feat_box_quanti[2:], radius)
                else:
                    tl_heatmaps[b,cls, feat_box_quanti[0],feat_box_quanti[1]] = 1 
                    br_heatmaps[b,cls, feat_box_quanti[2],feat_box_quanti[3]] = 1 
                tag_ind = tag_lens[b]
                tl_regrs[b,tag_ind, :] = reg_label[:2]
                br_regrs[b,tag_ind, :] = reg_label[2:]
                tl_inds[b, tag_ind] = feat_box_quanti[1] * train_params['output_size'][1] +feat_box_quanti[0] #xtl  can be seen as a way of encode
                br_inds[b, tag_ind] = feat_box_quanti[3] * train_params['output_size'][1] +feat_box_quanti[2] #xtl  can be seen as a way of encode
                tag_lens[b] += 1
          
        for b in range(self.batch_size):
            tag_len = tag_lens[b]
            tag_masks[b, :tag_len] = 1
        #if self.provide_label is None:
        #    self.provide_label = []
        #    for l in self.label_names:
        #        self.provide_label.append((l,eval(l).shape))
        
        label = []
        for l in self.label_names:
            label.append(mx.nd.array(eval(l)))
        return (data, label)

    def shutdown(self):
        if not self._shutdown:
            for _ in range(self._num_workers):
                self._key_queue.put((None,None))
            self._data_queue.put((None,None))
            self._shutdown = True

            
        
    def __del__(self):
        self.shutdown()
