import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.draw as draw
from skimage.util import invert
from skimage.morphology import erosion, selem
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.ndimage import distance_transform_edt

def plot_curv2d(im):
  # create figure
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  # add image
  cmap = copy.copy(plt.cm.get_cmap('jet'))
  cmap.set_bad(color='black')

  im = np.ma.masked_where(np.logical_and(im<0.0001,im>-0.0001), im)
  ax.imshow(im, cmap=cmap, interpolation='none')

  # hide axis
  plt.axis('off')
  # plt.savefig('./im/macular_hole_curv_2d.png', bbox_inches='tight', pad_inches=0)

  # show
  plt.show()

def plot2d(im):
  # create figure
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  # add image
  cmap = copy.copy(plt.cm.get_cmap('gray'))
  ax.imshow(im, cmap=cmap, interpolation='none')

  # hide axis
  plt.axis('off')
  # plt.savefig('./im/macular_hole_2d.png', bbox_inches='tight', pad_inches=0)

  # show
  plt.show()

def mean_curv2d(im, s=3, r=np.array([1,1])):
  # boundary
  ime = erosion(im, selem.ellipse(r[0], r[1]))
  imb = im.copy()
  imb[ime] = 0

  # distance - positive
  imdp = distance_transform_edt(invert(im), sampling=r)

  # distance - negative
  imdn = -1*distance_transform_edt(ime, sampling=r)

  # distance - full
  imd = imdn + imdp

  # mean curvature
  k = selem.ellipse(s*r[0], s*r[1])
  k = k/np.sum(k)
  imcurv = convolve(imd, k)
  imcurv[invert(imb)] = 0

  return imcurv

def gauss_curv2d(im, s=3, r=np.array([1,1])):
  # boundary
  ime = erosion(im, selem.ellipse(r[0], r[1]))
  imb = im.copy()
  imb[ime] = 0

  # distance - positive
  imdp = distance_transform_edt(invert(im), sampling=r)

  # distance - negative
  imdn = -1*distance_transform_edt(ime, sampling=r)

  # distance - full
  imd = imdn + imdp

  # gauss curvature
  imcurv = gaussian_filter(imd, r*s)
  imcurv[invert(imb)] = 0

  return imcurv

def disk_image2d():
  # image
  im = np.zeros((100, 100))==1

  # add object
  sx, sy = np.array(im.shape)/2
  rr, cc = draw.disk((sx, sy), sx/2, shape=im.shape)
  im[rr, cc] = 1
  rr, cc = draw.disk((sx/2, sy), sx/2, shape=im.shape)
  im[rr, cc] = 0

  return im

def rectangle_image2d():
  # image
  im = np.zeros((100, 100))==1

  # add object
  sx, sy = np.array(im.shape)
  rr, cc = draw.rectangle((int(sx/4), int(sy/4)), (int(3*sx/4), int(3*sy/4)), shape=im.shape)
  im[rr, cc] = 1
  rr, cc = draw.rectangle((int(sx/3), int(sy/3)), (int(2*sx/3), int(2*sy/3)), shape=im.shape)
  im[rr, cc] = 0

  return im

if __name__ == '__main__':

  # generate image
  # im = disk_image2d()
  # im = rectangle_image2d()

  # load image
  im = io.imread('./im/macular_hole_3d.tif')
  im = np.moveaxis(im, 0, -1)
  im = invert(im)
  z = 25
  im = im[:,:,z]

  # curvature
  # imcurv = gauss_curv2d(im)
  imcurv = mean_curv2d(im)

  # display
  plot2d(im)
  plot_curv2d(imcurv)
