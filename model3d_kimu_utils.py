# -*- coding: utf-8 -*-

import numpy as np
import skimage.color
import model3d_kimu as m3d
import m2m_distance as md
import os
import f2f_localaxis as f2f
import re
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def makedir(dir):

    if os.path.exists(dir) == False:
        os.mkdir(dir)
    return dir

def change_local(face_dist,affine):
    """face_dist 3x3 affine 4x4"""
    ones = np.array([1.0,1.0,1.0])
    fd = np.vstack((face_dist,ones))
    fd_t = affine.dot(fd)

    return fd_t[0:3,0:3]

"""""
    face to face の距離を求める
    mesh2 - mesh1 を行う
    m2m_distanceを作成
"""""

def defom_p(rigid_t,point):

    data = point.reshape([3,1])
    data = np.vstack((data,1))
    trans_p = rigid_t.dot(data.reshape([4,1]))
    trans_p = trans_p[0:3,:]
    return trans_p.reshape([3,1])

def deform_vertex(rigid_t, vertexes):
    row, col = vertexes.shape
    dM = np.empty([3, col])

    for verNum in xrange(col):
        data = vertexes[:, verNum]
        data = np.hstack((data, 1))
        tran_p = rigid_t.dot(data.reshape([4, 1]))
        tran_p = tran_p[0:3, :]
        dM[:, verNum] = tran_p.reshape([1, 3])

    return dM.reshape([3, col])

def load_ply_file(fname):
    with open(fname, 'r') as fin:
        num_vertex = 0
        num_face = 0
        read_header = True
        while read_header:
            line = fin.readline()
            line_s = line.split(' ')
            if line_s[0] == 'element':
                if line_s[1] == 'vertex':
                    num_vertex = int(line_s[2])
                elif line_s[1] == 'face':
                    num_face = int(line_s[2])
            elif line_s[0].strip() == 'end_header':
                read_header = False

        vertices = np.ndarray((3, num_vertex))

        for i in range(num_vertex):
            line = fin.readline()
            line_s = line.split(' ')
            vertices[:, i] = [float(line_s[0]), float(line_s[1]), float(line_s[2])]

        faces = np.ndarray((3, num_face), dtype=np.int)

        for i in range(num_face):
            line = fin.readline()
            line_s = line.split(' ')
            faces[:, i] = [int(line_s[1]), int(line_s[2]), int(line_s[3])]
        return vertices, faces

def save_ply_file(fname, v, f=None, lab=None, tar=None):
    cname = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
    cd = skimage.color.color_dict
    cols1 = [np.array(cd[cn]) * 255 for cn in cname]
    cols2 = [np.array(cd[cn]) * 127 for cn in cname]
    cols = cols1 + cols2

    if lab is not None:
        lset = set(list(lab[0, :]))
        ldic = {l: i for i, l in enumerate(lset)}

    n = v.shape[1]  # == c.shape[1]

    if f is not None:
        k = f.shape[1]
    else:
        k = 0
    with open(fname, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('comment author: Greg Turk\n')
        fout.write('comment object: another cube\n')
        fout.write('element vertex %d\n' % n)
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('element face %d\n' % k)
        fout.write('property list uchar int vertex_index\n')
        fout.write('end_header\n')

        for i in range(n):
            if lab is not None:
                c = cols[ldic[lab[0, i]] % len(cols)]
                if tar is not None and lab[0, i] != tar:
                    c = np.array((0,0,0))
            else:
                c = np.array((150, 150, 150))

            fout.write('%f %f %f %.0f %.0f %.0f\n' % (v[0, i], v[1, i], v[2, i], c[0], c[1], c[2]))

        for i in range(k):
            fout.write('3 %d %d %d\n' % (f[0, i], f[1, i], f[2, i]))

def save_ply_file_color(fname, v, f=None, colorList=None):

    n = v.shape[1]  # == c.shape[1]

    if f is not None:
        k = f.shape[1]
    else:
        k = 0
    with open(fname, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('comment author: Greg Turk\n')
        fout.write('comment object: another cube\n')
        fout.write('element vertex %d\n' % n)
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('element face %d\n' % k)
        fout.write('property list uchar int vertex_index\n')
        fout.write('end_header\n')


        colorList = colorList.transpose()

        for i in range(n):
            if colorList is None:
                c = np.array((150, 150, 150))
            else:
                c = colorList[i,:]

            fout.write('%f %f %f %.0f %.0f %.0f\n' % (v[0, i], v[1, i], v[2, i], c[0], c[1], c[2]))

        for i in range(k):
            fout.write('3 %d %d %d\n' % (f[0, i], f[1, i], f[2, i]))

def makeRed(num):
    red = np.zeros((3,num))
    red[0:,] = red[0,:] + 255
    return red

def makeYellow(num):
    y = np.zeros((3, num))
    y[0:, ] = y[0, :] + 255
    y[1:, ] = y[1, :] + 255

    return y

"""test data"""

def data():
    v = np.array([[0, 1, 0, 0]
                , [0, 0, 1, 0]
                , [0, 0, 0, 1]])

    f = np.array([[0, 0, 0],
                  [1, 1, 2],
                  [2, 3, 3]])
    return v,f

def data2():
    v = np.array([[-3, 1, 0, 0]
                , [0, 0, 1, 0]
                , [-1, 3, 2, 2]])

    f = np.array([[1, 0, 0],
                 [0, 1, 2],
                 [2, 3, 3]])
    return v,f

def data3():
    v = np.array([[0, 0, 0]
                , [0, 0, 1]
                , [0, -1, 0]])

    f = np.array([[0],
                  [1],
                  [2]])
    return v,f

def data4():
    v = np.array([[1, 1, 2]
                , [0, 0, 1]
                , [0, -1, 0]])

    f = np.array([[0],
                  [1],
                  [2]])
    return v,f

def data5():
    v = np.array([[0, 0, 0, 0]
                , [0, 0, 1, -1]
                , [0, -1, 0, 0]])

    f = np.array([[0, 0],
                  [1, 1],
                  [2, 3]])
    return v,f


def data6():
    v = np.array([[1, 1, 1, 1]
                , [0, 0, 1, -1]
                , [0, -1, 0, 0]])

    f = np.array([[0, 0],
                  [1, 1],
                  [2, 3]])
    return v,f

def red():
    c = np.array([[255, 0,  0,   0]
                 ,[0,   255,255,   0]
                 ,[0,   0,  0, 0]])
    return c

if __name__ == "__main__":

    v1,f1 = data3()
    v2,f2 = data4()

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()
    m1.input_data(v1,f1)
    m2.input_data(v2,f2)
    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()
    m2.calc_local_coord()
    m2.write_ply("deform_pre.ply")
    m2.check_local_axis(0,"local_test")

