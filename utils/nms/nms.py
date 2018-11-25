import numpy as np
#from .gpu_nms import gpu_nms
def nms(bboxes,threshold ,device_id):
    return gpu_nms(bboxes,threshold,device_id)
#def nms(bboxes,threshold = 0.7):
#    def IOU(box, bboxes):
#        w = np.maximum(np.minimum(box[2],bboxes[:,2]) - np.maximum(box[0],bboxes[:,0]) + 1,0)
#        h = np.maximum(np.minimum(box[3],bboxes[:,3]) - np.maximum(box[1],bboxes[:,1]) + 1,0)
#        return (h * w) / ((box[2] - box[0] + 1) * (box[3] - box[1] + 1) + (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1) - w * h)
#    if bboxes.size == 0:
#        return []
#    idx = bboxes[:,4].ravel().argsort()[::-1]
#
#    keep = []
#    while idx.size > 0:
#        keep.append(idx[0])
#        overlap = IOU(bboxes[idx[0],:4],bboxes[idx[1:],:4])
#        inds = np.where(overlap <= threshold)[0]
#        idx = idx[inds + 1]
#
#    return keep

    
# def nms(dets, thresh):
#     """
#     greedily select boxes with high confidence and overlap with current maximum <= thresh
#     rule out overlap >= thresh
#     :param dets: [[x1, y1, x2, y2 score]]
#     :param thresh: retain overlap < thresh
#     :return: indexes to keep
#     """
#     if dets.shape[0] == 0:
#         return []

#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]

#     return keep
def py_nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_nms(dets, thresh, max_dets = -1):
    if dets.shape[0] == 0:
        return []
#        return np.zeros((0, 5))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    if max_dets == -1:
        max_dets = order.size

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0

    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]
        dets[i, 4] = scores[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = rescore(ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]
    return keep
#    dets = dets[keep, :]
#    return dets

