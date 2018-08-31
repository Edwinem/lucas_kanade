import numpy as np
import cv2


def bilinear_interp(img, x_loc, y_loc):
    '''
    Does bilinear interpolation for a point in the image
    :param img: Image
    :param x_loc:
    :param y_loc:
    :return:
    '''
    xi = int(x_loc)
    yi = int(y_loc)
    top_left = img[yi][xi]
    top_right = img[yi + 1][xi]
    bot_left = img[yi][xi + 1]
    bot_right = img[yi + 1][xi + 1]
    I1 = (yi + 1 - y_loc) * top_left + (y_loc - yi) * top_right
    I2 = (yi + 1 - y_loc) * bot_left + (y_loc - yi) * bot_right
    val = (xi + 1 - x_loc) * I1 + (x_loc - xi) * I2
    return val


def symmetrix_skew(vec3):
    '''
    Creates a 3x3 symmetrix skew matrix for a vector x3. This can be used for instance as a cross product or in other
    applications
    :param vec3: Vector of size 3
    :return:
    '''
    output = np.zeros((3, 3))
    output[0, 0] = 0
    output[0, 1] = -vec3(2)
    output[0, 2] = vec3(1)
    output[1, 0] = vec3(2)
    output[1, 1] = 0
    output[1, 2] = -vec3(0)
    output[2, 0] = -vec3(1)
    output[2, 1] = vec3(0)
    output[2, 2] = 0
    return output


def transform_mat_2_twist(mat44):
    '''
    Converts a homogenous 4x4 transformation matrix to its twist form(vector x6)
    :param mat44:
    :return: Vector of size 6 representing the twist

    See lie algebra or rodrigues (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

    '''
    rot = mat44[0:3,0:3]

    theta = np.arccos(0.5 * (rot.trace() - 1))

    ret = np.zeros((6, 1))

    if (theta > 1e-10):
        s = np.sin(theta)
        c = np.cos(theta)
        a = s * (1.0 / theta)
        b = (1.0 - c) * (1.0 / (theta * theta))

        W = (theta / (2 * s)) * (rot - rot.transpose())
        V = np.eye(3) - 0.5 * W + (1 / (theta * theta)) * (1-(a / (2 * b))) * W * W
        ret[0:3] = (W(2, 1), W(0, 2), W(1, 0))

        ret_trans=ret[3:6]
        ret_trans=ret_trans.reshape(3,1)
        ret_trans = (V * mat44[3, 0:3]).reshape(3)
    else:
        ret_trans=ret[3:6]
        ret_trans=ret_trans.reshape(3,1)
        ret_trans = ( mat44[3, 0:3]).reshape(3)

    return ret

def twist_2_mat44(twist_vec):
    '''
    Transform a twist vector (x6) to a homogenous 4x4 Transformation matrix
    :param twist_vec:
    :return: 4x4 matrix

    See lie algebra or rodrigues
    '''
    ret=np.eye(4)

    theta = np.linalg.norm((twist_vec[0:3]))
    if(theta > 1e-8):
        a = np.sin(theta)
        b = 1.0 - np.cos(theta)
        t_i = 1.0 / theta
        S = t_i * symmetrix_skew( twist_vec[0:2] )
        S2 = S * S
        I = np.eye(3)

        ret[0:3,0:3] = I + a*S + b*S2
        ret[0:3,3] = (I + b*t_i*S + (theta - a)*t_i*S2) * twist_vec[3:6]

    else:
        ret[0:3,3][:]=twist_vec[3:6].reshape(3)


    return ret


def mat44_2_vec6(mat44):
    rot = mat44[0:3, 0:3]
    sy = np.sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0))

    out_vec = np.zeros((6, 1))

    if not sy < 1e-6:
        out_vec[0] = np.arctan2(rot(2, 1), rot(2, 2))
        out_vec[1] = np.arctan2(-rot(2, 0), sy)
        out_vec[2] = np.arctan2(rot(1, 0), rot(0, 0))
    else:
        out_vec[0] = np.arctan2(-rot(1, 2), rot(1, 1))
        out_vec[1] = np.arctan2(-rot(2, 0), sy)
        out_vec[2] = 0

    out_vec[0:3] = mat44[3, 0:3]
    return out_vec



def img2world(pts, K):
    pointcloud=np.copy(pts)

    pointcloud[:, 0]= (pointcloud[:, 0] - K[0, 2]) / K[0, 0]
    pointcloud[:, 1] = (pointcloud[:, 1] - K[1, 2]) / K[1, 1]
    pointcloud[:, 0]= pointcloud[:, 0] * pointcloud[:, 2]
    pointcloud[:, 1] = pointcloud[:, 1] * pointcloud[:, 2]
    return pointcloud

def world2img(pt,K):
    new_pt=np.copy(pt)
    new_pt=new_pt/new_pt[2]


def load_tum_depth(filename):
    '''
    Loads a tum depth image, and scales it to meters
    :param filename: Filepath to image in tum format. Is of type .png
    :return: A 2d image stored as floats with the units in meters

    TUM stores their depth images as uint16 and are scaled by 5000 (e.g 1meter=5000). So this functions takes care
    of loading the image and scaling it properly

    '''
    depth_img=cv2.imread(filename)
    depth_img=depth_img.astype(float)
    depth_img[:]=depth_img[:]/5000.0
    return depth_img


def TsukubaCameraK():
    return np.matrix([[615.0,0,320],[0,615.0,240],[0,0,1]])

<<<<<<< HEAD
def pyrdown_cam_matrix(K):
    '''
    Scales down a camera matrix (fx,fy,cx,cy)
    :param K: 3x3 camera matrix
    :return:

    See LSD-SLAM specifically this file
    https://github.com/tum-vision/lsd_slam/blob/master/lsd_slam_core/src/DataStructures/Frame.cpp


    '''
    new_K=np.eye(3)
    new_K[0,0]=K[0,0]*0.5 #fx
    new_K[1,1]=K[1,1]*0.5 #fy

    #The weird .5 comes from the fact that we assume pixel centers are at 0.5 not at (0,0)
    new_K[0,2]=K[0,2]+0.5/2-0.5  #cx
    new_K[1, 2] = K[1, 2] + 0.5 / 2 - 0.5  #cy
=======
def pyrdown_median(image):
    blurred=cv2.medianBlur(image,3)
    rows,cols=image.shape
    new_img=np.empty((int(rows/2),int(cols/2)),dtype=image.dtype)
    for r in range(0,new_img.shape[0]):
        for c in range(0,new_img.shape[1]):
            new_img[r][c]=blurred[r*2,c*2]
    return new_img
>>>>>>> a816fedf64cee49001c8514428cbc4773ab227c4

