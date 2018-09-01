import cv2
from lk import *
from warps import RBWarp
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



num_levels=3

ref_pyr=[ref_img]
cur_pyr=[cur_img]
ref_depth_pyr=[ref_depth]
Ks=[TsukubaCameraK()]
for level in range(0,num_levels-1):
    ref_pyr.append(cv2.pyrDown(ref_pyr[level]))
    cur_pyr.append(cv2.pyrDown(cur_pyr[level]))
    ref_depth_pyr.append(pyrdown_median(ref_depth_pyr[level]))
    Ks.append(pyrdown_cam_matrix(Ks[level]))


level=2


rows,cols=ref_depth_pyr[level].shape
ref_pts=np.empty((rows*cols,3))

index=0
for r in range(0,rows):
    for c in range(0,cols):
        d=ref_depth_pyr[level][r,c]
        if(d>1.5 and d<3.0):
            ref_pts[index]=c,r,d
            index=index+1
ref_pts=np.resize(ref_pts,(index,3))
ref_inten=np.empty((ref_pts.shape[0]))

for idx in range(0,ref_pts.shape[0]):
    r=int(ref_pts[idx][1])
    c=int(ref_pts[idx][0])
    ref_inten[idx]=ref_pyr[level][r,c]


lk_method=ForwardCompositional()
warp=RBWarp()
lk_method.run_level(ref_pts,ref_inten,ref_pyr[level],cur_pyr[level],warp,Ks[level])









