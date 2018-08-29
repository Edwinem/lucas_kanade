import numpy as np


class BaseCostFunction():
    def __init__(self):
        #something


class Huber(BaseCostFunction):
    def __init__(self):
        self.k=0


    def configure(self,scale):
        self.k=scale

    def compute(self,t):
        t_abs=np.abs(t)
        if(t_abs<self.k):
            return 1.0
        else:
            return self.k/t_abs