import cv2
from lk import *
from skimage.filters import sobel


fs=cv2.FileStorage("test_data/tsukuba_depth_L_00001.xml", cv2.FILE_STORAGE_READ)
fn=fs.getNode("depth")
ref_depth=fn.mat()
fs=cv2.FileStorage("test_data/tsukuba_depth_L_00030.xml", cv2.FILE_STORAGE_READ)
fn=fs.getNode("depth")

cur_depth=fn.mat()

ref_img=cv2.imread('test_data/tsukuba_rgb_1.png',cv2.IMREAD_GRAYSCALE)
cur_img=cv2.imread('test_data/tsukuba_rgb_30.png',cv2.IMREAD_GRAYSCALE)

#Depth is in cm so convert to m
ref_depth[:]=ref_depth[:]/100.0


rows,cols=ref_depth.shape

ref_pts=np.empty((rows*cols,3))

num_levels=3

ref_pyr=[ref_img]
cur_pyr=[cur_img]
ref_depth_pyr=[ref_depth]
for level in range(0,num_levels-1):
    ref_pyr.append(cv2.pyrDown(ref_pyr[level]))
    cur_pyr.append(cv2.pyrDown(cur_pyr[level]))
    ref_depth_pyr.append(pyrdown_median(ref_depth_pyr[level]))










