from PIL import Image
import numpy as np
import os
import cv2
import random
DEBUG = True
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_detections
def _resize_image(image, detections, size):
    print('1.1.1')
    detections    = detections.copy()
    print('1.1.2')
    height, width = image.shape[0:2]
    print('1.1.3')
    new_height, new_width = size

    print('1.1.4')
    print(image.shape,new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    print('1.1.5')

    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def crop_image(image, center, size):
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, 3), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset


def get_image(roidb,config):
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    processed_boxes = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
        print(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        detections = np.array(roi_rec['boxes']).astype(np.float64)
        if roidb[i]['flipped']:
            im = im[:,::-1,:]
        rand_scales = config['scales']
        input_size = config['input_size']
        output_size = config['output_size']
        border = config['border']

        if not DEBUG and config['rand_crop']:
            im, detections = random_crop(im, detections, rand_scales, input_size, border)
        print('im at 1.1')
        print('im:',im.shape,'det:',detections.shape,'input_size', input_size)
        im, detections = _resize_image(im, detections, input_size)
        print('im at 1.2')
        boxes = np.zeros((detections.shape[0],5))
        print('im at 1.3')
        boxes[:,:4] = detections
        print('im at 1.4')
        boxes[:,4] = roi_rec['gt_classes'] - 1
        print('im at 1.5')
        boxes  = _clip_detections(im, boxes)
        print('im at 1.6')
        if len(boxes) != 0 and boxes.max() > 510:
            import pdb
            pdb.set_trace()
            stop = 1

        print('im at 2')

        im = im.astype(np.float32)/255.
        if not DEBUG and config['rand_color']:
            color_jittering_(np.random.RandomState(123), im)
        normalize_(im, config['mean'], config['std'])
        im_tensor = im.transpose((2,0,1))[None,:,:,:]




        print('im at 3')
        processed_ims.append(im_tensor)
        processed_boxes.append(boxes)
    return processed_ims, processed_boxes

def get_test_image(roidb,config):
    num_images = len(roidb)
    processed_ims = []
    info_lst = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
        print(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)

        

        detections = np.array(roi_rec['boxes']).astype(np.float64)
        h, w = im.shape[0:2]
        scale = config['test_scales']
        new_height = int(h* scale)
        new_width = int(w* scale)
        new_center = np.array([new_height//2, new_width //2])
        inp_height = new_height | 127
        inp_width = new_width | 127
        images = np.zeros((1,3,inp_height,inp_width), dtype = np.float32)
        ratios = np.zeros((1,2),dtype = np.float32)
        borders = np.zeros((1,4),dtype = np.float32)
        resizes = np.zeros((1,2),dtype = np.float32)

        out_height, out_width = (inp_height +1 )//4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio = out_width / inp_width
        resized_image = cv2.resize(im,(new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])
        resized_image = resized_image / 255.
        normalize_(resized_image, config['mean'], config['std'])
        images[0] = resized_image.transpose((2,0,1))
        borders[0] = border
        resizes[0] = [int(h* scale), int(w* scale)]
        ratios[0] = [height_ratio, width_ratio]
        processed_ims.append(images)
        info_lst.append(np.concatenate([borders, resizes, ratios], axis = 1))

    return processed_ims, info_lst

