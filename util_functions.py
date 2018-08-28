import numpy as np


def bilinear_interp(img, x_loc, y_loc):
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


def mat44_2_vec6(mat44):
    rot = mat44[0:3, 0:3]
    sy = np.sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0))

    out_vec = np.zeros((6, 1))

    if not sy < 1e-6:
        out_vec[0] = np.atan2(rot(2, 1), rot(2, 2))
        out_vec[1] = np.atan2(-rot(2, 0), sy)
        out_vec[2] = np.atan2(rot(1, 0), rot(0, 0))
    else:
        out_vec[0] = np.atan2(-rot(1, 2), rot(1, 1))
        out_vec[1] = np.atan2(-rot(2, 0), sy)
        out_vec[2] = 0

    out_vec[0:3] = mat44[3, 0:3]
    return out_vec


def symmetrix_skew(vec3):
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
    rot = mat44[0:2.0:2];
    theta = np.acos(0.5 * (rot.trace() - 1));

    ret = np.zeros((6, 1))

    if (theta > 1e-10):
        s = np.sin(theta);
        c = np.cos(theta);
        a = s * (1.0 / theta);
        b = (1.0 - c) * (1.0 / (theta * theta))

        W = (theta / (2 * s)) * (R - R.transpose());
        V = np.eye(3) - 0.5 * W + (1 / (theta * theta)) * (1-(a / (2 * b))) * W * W;
        ret[0:2] = (W(2, 1), W(0, 2), W(1, 0));
        ret[3:5] = V * mat44[3, 0:2];
    else:
        ret[3:5] = mat44[3, 0:2];

    return ret;

def twist_2_mat44(twist_vec):
    ret=np.eye(4)

    theta = np.linalg.norm((twist_vec[0:3]))
    if(theta > 1e-8):
        a = np.sin(theta);
        b = 1.0 - np.cos(theta);
        t_i = 1.0 / theta;
        S = t_i * symmetrix_skew( twist_vec[0:2] )
        S2 = S * S;
        I = np.eye(3)

        ret[0:2,0:2] = I + a*S + b*S2;
        ret[0:2,3] = (I + b*t_i*S + (theta - a)*t_i*S2) * twist_vec[3:5]

    else:
        ret[0:2,3] = twist_vec[3:5]


    return ret;
