# -*- coding: utf-8 -*-

import model3d_kimu as m3d
import numpy as np

def data():
    v = np.array([[0, 1, 0, 0]
                , [0, 0, 1, 0]
                , [0, 0, 0, 1]])

    f = np.array([[0, 0, 0],
                 [1, 1, 2],
                 [2, 3, 3]])
    return v,f

def data2():
    v = np.array([[0, 1, 0]
                , [0, 0, 1]
                , [1, 1, 1]])

    f = np.array([[0],
                  [1],
                  [2]])
    return v,f

def test():

    t = m3d.model3d_plus()

    t.read_ply("./data/move/move00-shape.ply")

    t.calc_all()
    print "f,no",t.f_normals
    row,col = t.f_normals.shape
    print "num" ,col



def test2():
    t2 = m3d.model3d_changeLo()

    v,f = data2()

    t2.input_data(v,f)
    t2.calc_all()
    t2.write_ply("test_kimu.ply")

    print "normals"
    print t2.f_normals
    print "center"
    print t2.face_centers

    t2.calc_local_coord()

    """
    print "center"
    print t2.face_centers
    print "normals"
    print t2.f_normals
    t2.calc_normal2()
    print t2.f_normals

    print "face,cen"
    print t2.face_centers
    print "p_nomal"
    print t2.p_normals
    """




if __name__ == "__main__":
    test()
    #test2()
    #test3()
