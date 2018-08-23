

from abc import ABCMeta, abstractmethod
import numpy as np



class BaseWarp:
    __metaclass__ = ABCMeta

    @abstractmethod
    def warp_point(self,point): pass

    @abstractmethod
    def calc_warp_jac(self,point): pass

    @abstractmethod
    def update_params(self,update_vec): pass



class EuclideanWarp(BaseWarp):

    def __init__(self):
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
        #with respect to theta
        dXdtheta=-np.sin(self.theta)*point[0]+np.cos(self.theta)*point[1]
        dXdtx=1 #for both its 1
        dYdtheta=-np.cos(self.theta)*point[0]-np.sin(self.theta)*point[1]

        return np.matrix([[dXdtheta,dXdtx,0],[dYdtheta,0,dXdtx]])

