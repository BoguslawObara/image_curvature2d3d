import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import io
import skimage.draw as draw
from skimage.util import invert
from skimage.morphology import erosion
from skimage.measure import marching_cubes
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_curv3d(im, imw):
  # create figure
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')

  # extract mesh
  verts, faces, _, _ = marching_cubes(im, 0.5, spacing=[1,1,1], step_size=3)
  cent = verts[faces]
  cent = np.mean(cent, axis=1)
  xi = cent[:,0]
  yi = cent[:,1]
  zi = cent[:,2]
  x = np.linspace(0, im.shape[0], im.shape[0])
  y = np.linspace(0, im.shape[1], im.shape[1])
  z = np.linspace(0, im.shape[2], im.shape[2])
  curv = interpn((x,y,z), imw, np.array([xi,yi,zi]).T)
  norm = colors.Normalize(vmin=min(curv), vmax=max(curv), clip=True)
  cmap = plt.cm.get_cmap('jet')
  cmap = cmap(norm(curv))

  # add mesh
  mesh = Poly3DCollection(verts[faces])
  mesh.set_edgecolor('k')
  mesh.set_facecolor(cmap)
  ax.add_collection3d(mesh)
  ax.set_xlim(0, im.shape[0])
  ax.set_ylim(0, im.shape[1])
  ax.set_zlim(0, im.shape[2])

  # hide axis
  plt.axis('off')
  # plt.savefig('./im/macular_hole_curv_3d.png', bbox_inches='tight', pad_inches=0)

  # show
  plt.tight_layout()
  plt.show()

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
  # plt.savefig('./im/macular_hole_curv_3d_2d.png', bbox_inches='tight', pad_inches=0)

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
  # plt.savefig('./im/macular_hole_3d_2d.png', bbox_inches='tight', pad_inches=0)

  # show
  plt.show()

def ellipsoid(a, b, c):
  e = draw.ellipsoid(a, b, c).astype(int)
  return e[1:-1,1:-1,1:-1]

def mean_curv3d(im, s=3, r=np.array([1,1,1])):
  # boundary
  ime = erosion(im, ellipsoid(r[0], r[1], r[2]))
  imb = im.copy()
  imb[ime] = 0

  # distance - positive
  imdp = distance_transform_edt(invert(im), sampling=r)

  # distance - negative
  imdn = -1*distance_transform_edt(ime, sampling=r)

  # distance - full
  imd = imdn + imdp

  # mean curvature
  k = ellipsoid(s*r[0], s*r[1], s*r[2])
  k = k/np.sum(k)
  imcurv = convolve(imd, k)
  imcurv_b = imcurv
  imcurv_b[invert(imb)] = 0

  return imcurv_b, imcurv

def gauss_curv3d(im, s=3, r=np.array([1,1,1])):
  # boundary
  ime = erosion(im, ellipsoid(r[0], r[1], r[2]))
  imb = im.copy()
  imb[ime] = 0

  # distance - positive
  imdp = distance_transform_edt(invert(im), sampling=r)

  # distance - negative
  imdn = -1*distance_transform_edt(ime, sampling=r)

  # distance - full
  imd = imdn + imdp

  # mean curvature
  imcurv = gaussian_filter(imd, r*s)
  imcurv_b = imcurv
  imcurv_b[invert(imb)] = 0

  return imcurv_b, imcurv

def ellipsoid_image3d():
  # image
  im = np.zeros((100, 100, 100))==1

  # add object
  sx, sy, sz = np.array(im.shape)/2
  e = ellipsoid(sx/2, sx/2, sx/2)
  x, y, z = np.nonzero(e)
  im[x + int(sx/2), y + int(sy/2), z + int(sz/2)] = 1
  e = ellipsoid(sx/2, sx/2, sx/2)
  x, y, z = np.nonzero(e)
  im[x + int(sx/4), y + int(sy/2), z + int(sz/2)] = 0

  return im

def cube_image3d():
  # image
  im = np.zeros((100, 100, 100))==1

  # add object
  sx, sy, sz = np.array(im.shape)
  im[int(sx/4):int(3*sx/4), int(sy/4):int(3*sy/4), int(sz/4):int(3*sz/4)] = 1
  im[int(sx/3):int(2*sx/3), int(sy/3):int(2*sy/3), int(sz/3):int(2*sz/3)] = 0

  return im

if __name__ == '__main__':

  # generate image
  # im = cube_image3d()
  # im = ellipsoid_image3d()

  # load image
  im = io.imread('./im/macular_hole_3d.tif')
  im = np.moveaxis(im, 0, -1)
  im = invert(im)

  # curvature
  # imcurv, imcurv_f = gauss_curv3d(im, s=10, r=np.array([1,1,1/5]))
  imcurv, imcurv_f = mean_curv3d(im, s=10, r=np.array([1,1,1/5]))

  # display
  z = 25
  plot2d(im[:,:,z])
  plot_curv2d(imcurv[:,:,z])
  # plot_curv3d(im, imcurv_f)