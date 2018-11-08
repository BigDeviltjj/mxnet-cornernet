from __future__ import division
import numpy as np
import mxnet as mx
import cv2
from .utils import gaussian_radius, draw_gaussian
from .image import get_image


class DetRecordIter(mx.io.DataIter):
    def __init__(self,roidb, train_params,batch_size, shuffle
                 ):
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
        self.cur = 0
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
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
            self._get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data = self.data, label = self.label, 
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
        ims, roidbs = get_image(roidb,self.train_params)
        self.data =[mx.nd.array(np.concatenate(ims,axis = 0))]
        

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
        if self.provide_label is None:
            self.provide_label = []
            for l in self.label_names:
                self.provide_label.append((l,eval(l).shape))
        
        self.label = []
        for l in self.label_names:
            self.label.append(mx.nd.array(eval(l)))
        return True
            
        
