import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage import io
from skimage.transform import rescale

from lk import ForwardAdditive2D
from warps import EuclideanWarp

#Load image
image=io.imread('tum_img1.png',as_grey=True)

img=rescale(image,.5,mode='reflect')


#Our warp parameters
warp_theta=0
warp_tx=5
warp_ty=4

#Warp the original image to something else so we can solve for the warp
eucl_mat=np.matrix([[np.cos(warp_theta),np.sin(warp_theta),warp_tx],
                   [-np.sin(warp_theta),np.cos(warp_theta),warp_ty],
                   [0,0,1]]) # euclidean transform matrix
warped_img=tf.warp(img,np.linalg.inv(eucl_mat)) #skimage takes the inverse

#Create the type of warp we want to figure out
warp=EuclideanWarp()

#Lets solve it using the forward additive method
lk_method=ForwardAdditive2D()

#Pyramid version
#lk_method.run_lk(img,warped_img,warp)

lk_method.run_level(img,warped_img,warp,max_iter=20,show_debug=True)

print(warp)