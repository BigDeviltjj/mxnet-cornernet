import numpy as np
import mxnet as mx
import cv2
from .utils import gaussian_radius, draw_gaussian

DEBUG = True

class DetRecordIter(mx.io.DataIter):
    def __init__(self,train_params, path_imgrec, batch_size, data_shape, path_imglist='',
                 label_width = -1, label_pad_width = -1, label_pad_value = -1,
                 resize_mode = 'force', mean_pixels = [123.68, 116.779, 103.939],
                 std_pixels = [70,69,73]
                 ):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec = path_imgrec,
            path_imglist = path_imglist,
            label_width = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size = batch_size,
            data_shape = data_shape,
            mean_r = mean_pixels[0],
            mean_g = mean_pixels[1],
            mean_b = mean_pixels[2],
            std_r = std_pixels[0],
            std_g = std_pixels[1],
            std_b = std_pixels[2],
            rand_mirror_prob = 0.5,
            preprocess_threads = 16,
            resize_mode = resize_mode
        )
        self.train_params = train_params
        self.data_shape = data_shape
        self.provide_label = None
        self.label_names = ['tl_heatmaps','br_heatmaps','tl_regrs','br_regrs','tl_inds','br_inds', 'tag_masks']
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data

    def reset(self):
        return self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False
        
        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width

        
        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        image_size = np.array([1,self.data_shape[1]-1,self.data_shape[2]-1,self.data_shape[1]-1,self.data_shape[2]-1])
        label = np.where(label>-1,label * image_size[np.newaxis,np.newaxis,:],label)
#        label = np.where(label>-1,label * np.array((1,)+self.data_shape[1:] * 2)[np.newaxis,np.newaxis,:],label)

        train_params = self.train_params
        tl_heatmaps = np.zeros((self.batch_size, train_params['num_classes'],train_params['output_sizes'][0], train_params['output_sizes'][1]))
        br_heatmaps = np.zeros((self.batch_size, train_params['num_classes'],train_params['output_sizes'][0], train_params['output_sizes'][1]))
        tl_regrs = np.zeros((self.batch_size, train_params['max_tag_len'],2))
        br_regrs = np.zeros((self.batch_size, train_params['max_tag_len'],2))
        # tl_inds = np.zeros((2, self.batch_size * train_params['max_tag_len']))
        # br_inds = np.zeros((2, self.batch_size * train_params['max_tag_len']))
        tl_inds = np.zeros((self.batch_size, train_params['max_tag_len']))
        br_inds = np.zeros((self.batch_size, train_params['max_tag_len']))

        tag_masks = np.zeros((self.batch_size, train_params['max_tag_len']))
        tag_lens = np.zeros((self.batch_size,), dtype = np.int64)
        width_ratio = train_params['output_sizes'][0] / self.data_shape[2]
        height_ratio = train_params['output_sizes'][1] / self.data_shape[1]
        for b, single_label in enumerate(label):
            keep = np.where(single_label[:,0]>-1)[0]
            gt_boxes = single_label[keep]
            print("batch {}, box: ".format(b),gt_boxes)
            for gt_box in gt_boxes:

                box = gt_box[1:5]
                feat_box = box * np.array([width_ratio, height_ratio, width_ratio, height_ratio])
                feat_box_quanti = feat_box.astype(int)
                
                reg_label = feat_box - feat_box_quanti
                cls = int(gt_box[0])
                
                
                if train_params['gaussian_bump']:
                    width = np.ceil((gt_box[3] - gt_box[1])*width_ratio)
                    height = np.ceil(( gt_box[4] - gt_box[2])*height_ratio)
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
                tl_inds[b, tag_ind] = feat_box_quanti[1] * train_params['output_sizes'][1] +feat_box_quanti[0] #xtl  can be seen as a way of encode
                br_inds[b, tag_ind] = feat_box_quanti[3] * train_params['output_sizes'][1] +feat_box_quanti[2] #xtl  can be seen as a way of encode
                tag_lens[b] += 1
          
        for b in range(self.batch_size):
            tag_len = tag_lens[b]
            tag_masks[b, :tag_len] = 1
        if self.provide_label is None:
            self.provide_label = []
            for l in self.label_names:
                self.provide_label.append((l,eval(l).shape))
        
        self._batch.label = []
        for l in self.label_names:
            self._batch.label.append(mx.nd.array(eval(l)))
        return True
            
        
