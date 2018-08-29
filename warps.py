

from abc import ABCMeta, abstractmethod
import numpy as np
import util_functions as util



class BaseWarp:
    __metaclass__ = ABCMeta

    def __init__(self,jac_size):
        self.jac_size=jac_size


    @abstractmethod
    def warp_point(self,point): pass

    @abstractmethod
    def warp_point(self,point,K): pass

    @abstractmethod
    def calc_warp_jac(self,point): pass

    @abstractmethod
    def update_params(self,update_vec): pass





class EuclideanWarp(BaseWarp):
    r"""
    Class for a 2d Euclidean Warp.

    Is parameterized by tx,ty and theta

    Matrix form is

    cos(theta)  -sin(theta)  tx
    sin(theta)   cos(theta)  ty


    """

    def __init__(self):
        super().__init__(self,3)
        self.theta=0
        self.tx=0
        self.ty=0

    def set_params(self,theta,tx,ty):
        self.theta=theta
        self.tx=tx
        self.ty=ty

    def warp_point(self,point):
        x=np.cos(self.theta)*point[0]+np.sin(self.theta)*point[1]+self.tx
        y = -np.sin(self.theta) * point[0] + np.cos(self.theta) * point[1] + self.ty
        return np.array([x,y])

    def calc_warp_jac(self,point):
        '''

        :param point: A 2d point with x and y coordinates
        :return: A 2x3 jacobian matrix
        | dx
        |


        '''
        #with respect to theta
        dXdtheta=-np.sin(self.theta)*point[0]+np.cos(self.theta)*point[1]
        dXdtx=1 #for both its 1
        dYdtheta=-np.cos(self.theta)*point[0]-np.sin(self.theta)*point[1]

        return np.matrix([[dXdtheta,dXdtx,0],[dYdtheta,0,dXdtx]])

    def update_params(self,update_vec):
        self.theta=self.theta+update_vec[0]
        self.tx=self.tx+update_vec[1]
        self.ty=self.ty+update_vec[2]



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

    def update_params(self, update_vec):
        self.theta = self.theta + update_vec[0]
        self.tx = self.tx + update_vec[1]
        self.ty = self.ty + update_vec[2]

