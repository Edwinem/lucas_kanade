from util_functions import *
import numpy as np
from skimage import filters



class LKMethod:
    def __init__(self,  eps=10**-10):
        self.eps = eps
        self.max_iter=10

    def run_level(self,ref_img,ref_img_pts,cur_img,K,num_levels,initial_warp,show_debug=False):
        jac_size=initial_warp.jac_size

        H=np.zeros((jac_size,jac_size))
        Jgrad=np.zeros((jac_size,1))

        rows,cols=cur_img.shape()

        residuals=[]
        num_pts=ref_img_pts.shape()[0]
        pts_valid=np.ones((num_pts))

        grad_imx = filters.sobel_v(warped_image)
        grad_imy = filters.sobel_h(warped_image)


        for iter in range(0,self.max_iter):
            H.fill(0)
            Jgrad.fill(0)

            #First lets warp the 3d points to calculate the error

            warped_img_pts=initial_warp.warp_img_pts(ref_img_pts,K)

            for idx in range(0,num_pts):
                pt=warped_img_pts[idx]
                if(pt[0]>2 and pt[1]>2 and pt[0]<cols-2 and pt[1]<rows-2):
                    diff=

            #Calculate the warp Jacobian
            warp_jacs=initial_warp.calc_jacs(ref_img_pts,K)













def LukasKanade(orig_imag, warped_image, warp, max_iterations=10, show_dbg_img=False):
    grad_imx = filters.sobel_v(warped_image)
    grad_imy = filters.sobel_h(warped_image)



    for idx in range(0,max_iterations):
        print ('Iteration ' +str(idx))
        sum_err=0
        rows,cols=orig_imag.shape
        total_err2=0
        hessian=np.zeros((3,3))
        gngrad=np.zeros((3,1))
        num_obs=0
        for r in range(0,rows):
            for c in range(0,cols):
                point=np.array([c,r])#x and y so x as columns is first
                new_point=warp.warp_point(point)
                if(new_point[0]<0 or new_point[1]<0 or new_point[0]>=cols-1 or new_point[1]>=rows-1):
                    continue
                gradx=bilinear_interp(grad_imx,new_point[0],new_point[1])
                grady=bilinear_interp(grad_imy,new_point[0],new_point[1])
                intensity=bilinear_interp(warped_image,new_point[0],new_point[1])
                if(intensity==0): #bit of a hack. This is to avoid the areas with no actual image
                    continue
                jac_warp=warp.calc_warp_jacobian(point)
                jac_I=np.matrix([[gradx,grady]])

                jac=jac_I*jac_warp

                res=intensity-orig_imag[r][c]
                res2=res*res
                total_err2=total_err2+res2

                gngrad=gngrad+jac.transpose()*res
                hessian=hessian+jac.transpose()*jac
                num_obs=num_obs+1

        print('Average Error ', str(total_err2 / num_obs))
        H = hessian
        b = gngrad
        update = np.dot(np.linalg.inv(H), b)
        print('Delta update is ')
        print(update)
        update = update.flatten()
        warp.update_params(-update[0, 0], -update[0, 1], -update[0, 2])
        print()
        print('Updated warp')
        print(warp)

        if show_dbg_img:
            partly_warped = tf.warp(img, warp.inv_matrix())
            diff = warped_image - partly_warped
            squarer = lambda t: np.sqrt(t ** 2)
            vfunc = np.vectorize(squarer)
            diff = vfunc(diff)
            plt.imshow(diff, cmap="gray")