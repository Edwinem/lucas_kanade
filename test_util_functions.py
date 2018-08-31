import pytest
import util_functions as util
import numpy as np
import copy


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


def test_image_projection():
    pts=np.ones((5,3))

    K=np.eye(3)
    old_pts=copy.deepcopy(pts)

    new_pts=util.img2world(pts,K)

    assert (old_pts == new_pts).all()

    pts[0][2]=.5
    new_pts=util.img2world(pts,K)
    assert (new_pts[0][0]==0.5)




