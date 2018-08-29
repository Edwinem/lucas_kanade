import pytest
import util_functions as util
import numpy as np


def test_transform_2_twist():
    #Identity first
    mat=np.eye(4)
    twist=util.transform_mat_2_twist(mat)
    ans=np.zeros((6,1))
    assert (ans==twist).all()



def test_twist_2_transform():
    #Identity
    vec=np.zeros((6,1))
    mat=util.twist_2_mat44(vec)
    ans=np.eye(4)
    assert (ans==mat).all()
