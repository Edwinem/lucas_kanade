

from abc import ABCMeta, abstractmethod
import numpy as np
import util_functions as util

from skimage import transform as tf



class BaseWarp:
    __metaclass__ = ABCMeta

    def __init__(self,jac_size):
        self.jac_size=jac_size # The size of the jacobian which dictates the size of the Hessian

    #Warp a 2d point
    @abstractmethod
    def warp_point(self,point): pass

    #Warp multiple 2d points
    @abstractmethod
    def warp_points(self,points): pass

    #Warp a 3d point
    @abstractmethod
    def warp_point3d(self,point,K): pass

    #Warp a 3d point
    @abstractmethod
    def warp_pointcloud(self,point,K): pass

    #Calculate the jacobian of the warp of a single point
    @abstractmethod
    def calc_warp_jac(self,point): pass

    #Calculate the jacobian of the warp of multiple points
    @abstractmethod
    def calc_warp_jacs(self,points): pass

    #Calculate the jacobian of the warp of multiple points
    @abstractmethod
    def calc_warp3d_jacs(self,pointcloud): pass

    #Update the parameters using the forward additive method
    @abstractmethod
    def update_additive(self, update_vec): pass

    #Update the parameters using the forward compositional
    @abstractmethod
    def update_fcompositional(self, update_vec): pass

    #Update the parameters using the inveserse compositional
    @abstractmethod
    def update_inv_compositional(self, update_vec): pass

    @abstractmethod
    def print_warp(self): pass

    @abstractmethod
    def warp_image(self,image): pass

    @abstractmethod
    def warp_image_depth(self,image,depth_img): pass






class EuclideanWarp(BaseWarp):
    r"""
    Class for a 2d Euclidean Warp.

    Is parameterized by tx,ty and theta

    Matrix form is

    cos(theta)  -sin(theta)  tx
    sin(theta)   cos(theta)  ty


    """

    def __init__(self):
        super().__init__(3)
        self.theta=0
        self.tx=0
        self.ty=0

    def set_params(self,theta,tx,ty):
        '''
        Sets the values
        :param theta:
        :param tx: translation in the x
        :param ty: translation in the y
        :return:
        '''
        self.theta=theta
        self.tx=tx
        self.ty=ty

    def set_via_matrix(self,matrix):
        '''
        Set the parameters theta,tx,ty via a matrix
        :param matrix: A 3x3 matrix
        :return:
        '''
        self.tx=matrix[0][2]
        self.ty=matrix[1][2]
        self.theta=np.arccos(matrix[0][0])


    def warp_point(self,point):
        '''
        Simple Euclidean warp
        :param point: A 2d point
        :return:
        '''
        x=np.cos(self.theta)*point[0]+np.sin(self.theta)*point[1]+self.tx
        y = -np.sin(self.theta) * point[0] + np.cos(self.theta) * point[1] + self.ty
        return np.array([x,y])

    def warp_points(self,points):
        '''
        Warps a bunch of points using a euclidean warp
        :param points: List or matrix of pts. Should be size nx2
        :return: A matrix containing the warped pts in from nx2
        '''
        warped_pts=np.empty((points.shape[0],2))
        for idx in range(0,points.shape[0]):
            pt=points[idx]
            warped_pts[idx][0]=np.cos(self.theta)*pt[0]+np.sin(self.theta)*pt[1]+self.tx
            warped_pts[idx][1] = -np.sin(self.theta) * pt[0] + np.cos(self.theta) * pt[1] + self.ty
        return warped_pts


    def calc_warp_jac(self,point):
        '''
        Calculates the jacobian of the warp for a point

        The jacobian is of the form


           theta    tx  ty

        x   -s*x+c*y  1  0
        y   -cx-s*y   0  1

        c=cos(theta)
        s=sin(theta)

        :param point:
        :return: A np matrix with the data shown above
        '''
        dXdtheta=-np.sin(self.theta)*point[0]+np.cos(self.theta)*point[1]
        dXdtx=1 #for both its 1
        dYdtheta=-np.cos(self.theta)*point[0]-np.sin(self.theta)*point[1]

        return np.matrix([[dXdtheta,dXdtx,0],[dYdtheta,0,dXdtx]])

    def calc_warp_jacs(self,points):
        '''
        Calculates the jacobian
        :param points: A nx2 list/matrix of points
        :return: A list containg a matrix of the jacobian for the corresponding point
        '''
        dW_dp=[]

        for idx in range(0,points.shape[0]):
            pt=points[idx]
            dXdtheta=-np.sin(self.theta)*pt[0]+np.cos(self.theta)*pt[1]
            dXdtx=1 #for both its 1
            dYdtheta=-np.cos(self.theta)*pt[0]-np.sin(self.theta)*pt[1]
            dW_dp.append(np.matrix([[dXdtheta,dXdtx,0],[dYdtheta,0,dXdtx]]))
        return dW_dp


    def matrix(self):
        '''
        Get warp in Matrix form
        '''
        return np.matrix([[np.cos(self.theta),np.sin(self.theta),self.tx],
                          [-np.sin(self.theta),np.cos(self.theta),self.ty],
                          [0,0,1]])

    def inv_matrix(self):
        '''
        Get the inverted matrix form.
        '''
        mat=self.matrix()
        return np.linalg.inv(mat)

    def update_additive(self, update_vec):
        self.theta=self.theta+update_vec[0,0]
        self.tx=self.tx+update_vec[1,0]
        self.ty=self.ty+update_vec[2,0]

    def update_fcompositional(self, update_vec):
        warp_delta=EuclideanWarp
        warp_delta.set_params(update_vec[0, 0],update_vec[1, 0],update_vec[2, 0])
        self.theta = self.theta + update_vec[0, 0]
        self.tx = self.tx + update_vec[1, 0]
        self.ty = self.ty + update_vec[2, 0]
        mat=self.matrix()*warp_delta.matrix()

    def warp_image(self, image):
        return tf.warp(image, self.inv_matrix())

    def __str__(self):
        return "Parameters: theta is %s, tx is %s, ty is %s" % (self.theta, self.tx, self.ty)



class RBWarpEuler321(BaseWarp):
    r"""
    Class for a 2d Euclidean Warp.

    Is parameterized by tx,ty and theta

    Matrix form is

    cos(theta)  -sin(theta)  tx
    sin(theta)   cos(theta)  ty


    """

    def __init__(self):
        super().__init__(self, 6)
        self.rot=np.zeros((3,3))
        self.trans=np.zeros((3,1))


    def set_params(self,twist):
        mat=util.twist_2_mat44(twist)
        self.rot=mat[0:3,0:3]
        self.trans=mat[3,0:3]


    def warp_point(self, point,K):
        x = np.cos(self.theta) * point[0] + np.sin(self.theta) * point[1] + self.tx
        y = -np.sin(self.theta) * point[0] + np.cos(self.theta) * point[1] + self.ty
        return np.array([x, y])

    def calc_warp_jac(self, point):
        '''

        :param point: A 2d point with x and y coordinates
        :return: A 2x3 jacobian matrix
        | dx
        |


        '''
        # with respect to theta
        dXdtheta = -np.sin(self.theta) * point[0] + np.cos(self.theta) * point[1]
        dXdtx = 1  # for both its 1
        dYdtheta = -np.cos(self.theta) * point[0] - np.sin(self.theta) * point[1]

        return np.matrix([[dXdtheta, dXdtx, 0], [dYdtheta, 0, dXdtx]])

    def update_additive(self, update_vec):
        self.theta = self.theta + update_vec[0]
        self.tx = self.tx + update_vec[1]
        self.ty = self.ty + update_vec[2]

