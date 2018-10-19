from __future__ import division
import numpy as np

def gaussian_radius(det_size, overlap):

  w,h = det_size
  iou = overlap

  a1 = 4
  b1 = -2*(w+h)
  c1 = w*h*(1-iou)
  r1 = (-b1 - np.sqrt(b1**2 - 4 * a1 * c1))/(2*a1)

  a2 = 1
  b2 = -(w+h)
  c2 = w*h*(1-iou)/(1+iou)
  r2 = (-b2 - np.sqrt(b2**2 - 4 * a2 * c2))/(2*a2)
  
  a3 = 4*iou
  b3 = 2*(w+h)*iou
  c3 = w*h*(iou - 1)
  r3 = (-b3 + np.sqrt(b3**2 - 4 * a3 * c3))/(2*a3)
  return min(r1,r2,r3)

def gaussian_kernel(radius, mean, std):
  D = 2*radius + 1
  xx, yy = np.meshgrid(range(D),range(D))
  ret = np.exp(-((xx-radius)**2 + (yy-radius)**2)/(2*std**2))
  return ret
  

def draw_gaussian(heatmap, center, radius):
  k = gaussian_kernel(radius,mean = 0, std = (2 * radius + 1)/6)
  ctx, cty = center
  l = min(ctx, radius)
  r = min(radius + 1, heatmap.shape[1] - ctx)
  t = min(cty, radius)
  d = min(radius + 1, heatmap.shape[0] - cty)
  k = k[radius - t:radius + d, radius - l: radius + r]
  heatmap[cty - t:cty + d, ctx - l:ctx + r] = np.where(heatmap[cty - t:cty + d, ctx - l:ctx + r] > k, heatmap[cty - t:cty + d, ctx - l:ctx + r], k)
