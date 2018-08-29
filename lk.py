from util_functions import *
import numpy as np
from skimage import filters
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt
import copy


class LKMethod:
    def __init__(self, eps=10 ** -10):
        self.eps = eps
        self.max_iter = 10
        self.buffer=1

    def run_lk(self, ref_img, cur_img, initial_warp, num_levels=3, num_iter=[20, 20, 20], show_debug=False):

        iterations = copy.deepcopy(num_iter)
        if (num_levels != len(num_iter)):
            print('Warning num of levels does not match length of num iterations list. Will resort to default')
            for idx in range(0, num_levels):
                iterations.append(20)

        ref_pyr = tuple(pyramid_gaussian(ref_img, max_layer=num_levels, downscale=2))
        cur_pyr = tuple(pyramid_gaussian(cur_img, max_layer=num_levels, downscale=2))

        level = num_levels - 1
        while level >= 0:
            self.run_level(ref_pyr[level], cur_pyr[level], initial_warp, iterations[level], show_debug)
            level = level - 1


class ForwardAdditive2D(LKMethod):



    def run_level(self, ref_img, cur_img, initial_warp, max_iter=20, show_debug=False):

        img_object=0
        if(show_debug):
            img_object=plt.imshow(ref_img,cmap="gray")

        jac_size = initial_warp.jac_size

        #Hessian
        H = np.zeros((jac_size, jac_size))
        #Jacobian*gradient
        Jgrad = np.zeros((jac_size, 1))

        rows, cols = cur_img.shape

        residuals = []

        grad_imx = filters.sobel_v(cur_img)
        grad_imy = filters.sobel_h(cur_img)

        # Preallocating arrays
        # Stored as  | x1, x2, ...
        #            | y1, y2, ...
        pts = np.empty((rows * cols, 2))    # Points in the original image
        grads = np.empty((rows * cols, 2))  # gradients

        valid = np.ones((rows * cols))      #validity vector of pts 1 is valid 0 is invalid
        intensities = np.empty((rows * cols)) #vector which stores the intensities of the warped pts

        #Populate the original pts. They are just all the pts in an image
        index = 0
        for r in range(0, rows-1):
            for c in range(0, cols-1):
                pts[index][0] = c
                pts[index][1] = r
                index = index + 1

        prev_err = 200000000
        for iter in range(0, max_iter):
            print('On iteration {}'.format(iter))

            #Reset the various matrices and vectors
            H.fill(0)
            Jgrad.fill(0)
            valid.fill(1)
            total_err2 = 0
            num_valid = 0

            #Warp the pts with the given estimate
            new_pts = initial_warp.warp_points(pts)


            #For every warped pt. Check if it ends up in the image.
            for idx in range(0, new_pts.shape[0]):
                pt = new_pts[idx]
                if (pt[0] < self.buffer or pt[1] < self.buffer or pt[0] >= cols - self.buffer or pt[1] >= rows - self.buffer):
                    valid[idx] = 0
                else:
                    #Position is valid lets check the intensity
                    intensities[idx] = bilinear_interp(cur_img, pt[0], pt[1])
                    if (intensities[idx] == 0): #This removes any artifcats we may have due to the warp function
                        valid[idx] = 0
                        continue
                    #Pt is valid lets calculate the gradients in the cur image
                    grads[idx][0] = bilinear_interp(grad_imx, pt[0], pt[1])
                    grads[idx][1] = bilinear_interp(grad_imy, pt[0], pt[1])

            #Calculate the warp jacobian
            dW_dp = initial_warp.calc_warp_jacs(pts)

            #Here we build the linear system. A point is only added if it is valid
            for idx in range(0, len(valid)):
                if valid[idx] == 0:
                    continue
                # full jacobian dI*dW
                jac = grads[idx] * dW_dp[idx]
                jac=jac.transpose()
                r=int(pts[idx][1])
                c=int(pts[idx][0])
                res = ref_img[r][c] -intensities[idx] #residual
                total_err2 = total_err2 + res * res

                Jgrad = Jgrad + jac * res
                #Jgrad=Jgrad*-1
                H = H + jac * jac.transpose()
                num_valid = num_valid + 1

            #If the error is greater then before then stop iterating
            avg_err=total_err2/num_valid
            if (avg_err > prev_err):
                return False
            prev_err = avg_err

            #Solve the linear system
            update = np.dot(np.linalg.inv(H), Jgrad)
            initial_warp.update_additive(update)

            #If the update is less than eps then we end the run
            if (np.linalg.norm(update) < self.eps):
                print("Stopping due to dx/dt is too small")
                return True

            if show_debug:
                #Here we can visualize the debug images
                partly_warped = initial_warp.warp_image(ref_img)
                diff = cur_img - partly_warped
                squarer = lambda t: np.sqrt(t ** 2)
                vfunc = np.vectorize(squarer)
                diff = vfunc(diff)
                img_object.set_data(diff)
                plt.pause(.1)
                plt.draw()


###EVERYTHING BELOW THIS LINE IS WORK IN PROGRESS



class ForwardCompositional(LKMethod):



    def run_level(self, ref_img_pts,ref_intensities, ref_img,cur_img, initial_warp,K, max_iter=20, show_debug=False):
        assert (ref_img_pts.shape[1]==3)

        assert (ref_img_pts.shape[0]==ref_intensities.shape[1] or ref_img_pts.shape[0]==ref_intensities.shape[0])

        jac_size = initial_warp.jac_size

        #Hessian
        H = np.zeros((jac_size, jac_size))
        #Jacobian*gradient
        Jgrad = np.zeros((jac_size, 1))

        rows, cols = cur_img.shape


        #Build the pointcloud that we use to warp the pts
        pointcloud=img2world(ref_img_pts,K)


        grads = np.empty((ref_img_pts.shape[0], 2))  # gradients
        valid = np.ones((ref_img_pts.shape[0]))      #validity vector of pts 1 is valid 0 is invalid
        cur_intensities = np.empty((ref_img_pts.shape[0]))


        grad_imx = filters.sobel_v(cur_img)
        grad_imy = filters.sobel_h(cur_img)

        #In forward compositional this can be precomputed
        dW_dp = initial_warp.calc_warp3d_jacs(pointcloud,K)



        prev_err = 200000000
        for iter in range(0, max_iter):
            print('On iteration {}'.format(iter))

            #Reset the various matrices and vectors
            H.fill(0)
            Jgrad.fill(0)
            valid.fill(1)
            total_err2 = 0
            num_valid = 0

            #Warp the pts with the given estimate
            new_pts = initial_warp.warp_pointcloud(pointcloud)


            #For every warped pt. Check if it ends up in the image.
            for idx in range(0, new_pts.shape[0]):
                pt = new_pts[idx]
                if (pt[0] < self.buffer or pt[1] < self.buffer or pt[0] >= cols - self.buffer or pt[1] >= rows - self.buffer):
                    valid[idx] = 0
                else:
                    #Position is valid lets check the intensity
                    cur_intensities[idx] = bilinear_interp(cur_img, pt[0], pt[1])
                    #Pt is valid lets calculate the gradients in the cur image
                    grads[idx][0] = bilinear_interp(grad_imx, pt[0], pt[1])
                    grads[idx][1] = bilinear_interp(grad_imy, pt[0], pt[1])


            #Here we build the linear system. A point is only added if it is valid
            for idx in range(0, len(valid)):
                if valid[idx] == 0:
                    continue
                # full jacobian dI*dW
                jac = grads[idx] * dW_dp[idx]
                jac=jac.transpose()
                res = cur_intensities [idx]-ref_intensities[idx] #residual
                total_err2 = total_err2 + res * res

                Jgrad = Jgrad - jac * res
                H = H + jac * jac.transpose()
                num_valid = num_valid + 1

            #If the error is greater then before then stop iterating
            avg_err=total_err2/num_valid
            if (avg_err > prev_err):
                return False
            prev_err = avg_err

            #Solve the linear system
            update = np.dot(np.linalg.inv(H), Jgrad)
            initial_warp.update_additive(update)

            #If the update is less than eps then we end the run
            if (np.linalg.norm(update) < self.eps):
                print("Stopping due to dx/dt is too small")
                return True

            if show_debug:
                #Here we can visualize the debug images
                partly_warped = initial_warp.warp_image(ref_img)
                diff = cur_img - partly_warped
                squarer = lambda t: np.sqrt(t ** 2)
                vfunc = np.vectorize(squarer)
                diff = vfunc(diff)
                plt.imshow(diff, cmap="gray")
                plt.show(block=False)
                plt.pause(2)
                plt.close()


