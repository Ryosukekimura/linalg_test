# -*- coding: utf-8 -*-

import numpy as np
import skimage.color
import model3d_kimu as m3d
import m2m_distance as md
import os
import f2f_localaxis as f2f

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
    v = np.array([[0, 1, 0]
                , [0, 0, 1]
                , [0, 0, 0]])

    f = np.array([[0],
                  [1],
                  [2]])
    return v,f

def data4():
    v = np.array([[0, 1, 5]
                , [0, 2, 1]
                , [1, 0, 0]])

    f = np.array([[0],
                  [1],
                  [2]])
    return v,f

def red():
    c = np.array([[255, 0,  0,   0]
                 ,[0,   255,255,   0]
                 ,[0,   0,  0, 0]])
    return c

def test1():

    v1,f1 = data()
    v2,f2 = data2()
    c1 = red()

    m1 = m3d.model3d_changeLo()
    m1.input_data(v1,f1)
    m1.write_ply("m1.ply")

    m2 = m3d.model3d_changeLo()
    m2.input_data(v2,f2)
    m2.write_ply("m2.ply")

    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()


    dist = md.calc_f2f_distance_in_localcoord(m1, m2)
    # dist = md.calc_f2f_distance(m2, m1)
    colorlist,colorlist3 = md.distancelist2colorlist(dist)


    nv,nf = md.divide_face(m1)

    f0 = colorlist3[0]
    f0z = f0[2]
    print f0z
    save_ply_file_color("dist_color_test.ply",nv,nf,f0z)

def test2():
    out = "./colors/"
    if os.path.exists(out) == False:
        os.mkdir(out)

    for z in xrange(14):

        m1 = m3d.model3d_changeLo()
        m2 = m3d.model3d_changeLo()

        m1.read_ply("./data/move/move%02d-shape.ply" % z)
        m2.read_ply("./data/move/move%02d-vert.ply" % z)

        m1.calc_all()
        m2.calc_all()
        m1.calc_local_coord()

        #dist,dist_r = f2f.calc_one_frame(m1,m2)
        dist = md.calc_f2f_distance_in_localcoord(m1,m2)
        #dist = md.calc_f2f_distance(m2, m1)
        c1, c2 = md.distancelist2colorlist(dist)

        nv, nf = md.divide_face(m1)

        frame = out + "frame%02d/" % z
        if os.path.exists(frame) == False:
            os.mkdir(frame)

        for x in xrange(3):
            f = c2[x]

            outdir = frame + "distDataface%02d/" % x
            if os.path.exists(outdir) == False:
                os.mkdir(outdir)

            xyz = ["x", "y", "z"]

            for y in xrange(3):

                f2 = f[y]
                name = outdir + "%02d" % z + "dist_color_f%02d" % x + xyz[y] + ".ply"
                print name
                save_ply_file_color(name,nv,nf,f2)
                #np.savetxt(outdir + "dist_color_f"+xyz[y] + ".txt",f2,fmt='%.10f', delimiter=',')

    f0 = c2[2]
    f0x = f0[0]


    #save_ply_file_color("dist_color_test.ply",nv,nf,f0x)

if __name__ == "__main__":

    test2()