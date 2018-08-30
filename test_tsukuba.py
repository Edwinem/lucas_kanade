import cv2
from lk import *


fs=cv2.FileStorage("test_data/tsukuba_depth_L_00001.xml", cv2.FILE_STORAGE_READ)
fn=fs.getNode("depth")
ref_depth=fn.mat()
fs=cv2.FileStorage("test_data/tsukuba_depth_L_00030.xml", cv2.FILE_STORAGE_READ)
fn=fs.getNode("depth")

cur_depth=fn.mat()

ref_img=cv2.imread('test_data/tsukuba_rgb_1.png',cv2.IMREAD_GRAYSCALE)
cur_img=cv2.imread('test_data/tsukuba_rgb_30.png',cv2.IMREAD_GRAYSCALE)







