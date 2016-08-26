# -*- coding: utf-8 -*-

import numpy as np
import affine_me as affine
import tenbo_utils as tu
import get_nearplane_index as gni

#基本3dmodelクラス
class model3d:
    def __init__(self):
        self.face_flag = False
        self.vertex_flag = False
        self.vertexes = np.empty(1)
        self.faces = np.empty(1)
        self.ver_num = 0
        self.face_num = 0

    def input_data(self,vers,faces):
        self.vertexes = vers
        self.faces = faces

        row,col = self.vertexes.shape
        self.ver_num = col
        row,col = self.faces.shape
        self.face_num = col

        if self.vertexes.size > 0:
            self.vertex_flag = True
        if self.faces.size > 0:
            self.face_flag = True


#法線計算や重心計算などを加えたクラス
class model3d_plus(model3d):
    def __init__(self):
        self.f_normals = np.empty(1) #フェイスの法線を格納
        self.center = np.empty(1) #モデルの重心を格納
        self.face_centers = np.empty(1) #フェイスごとの重心を格納
        self.p_normals = np.empty(1) #点ごとの法線を格納

        self.normal_flag = False
        self.face_centers_flag = False

    def read_ply(self,fileName):
        """ply読み込み ＊要 tenbo_utils"""
        v, f = tu.load_ply_file(fileName)
        self.input_data(v, f)

    def write_ply(self,name):
        tu.save_ply_file(name,self.vertexes,self.faces)

    def calc_normal(self):
        """faceごとの法線"""
        if self.face_flag == True:
            ab = self.vertexes[:, self.faces[1, :]] - self.vertexes[:, self.faces[0, :]]
            ac = self.vertexes[:, self.faces[2, :]] - self.vertexes[:, self.faces[0, :]]

            C = np.cross(ab.transpose(),ac.transpose())
            self.f_normals = C.transpose()
            return self.f_normals
        return -1

    def calc_normal2(self):
        #法線ベクトルの正規化を行っている
        if self.face_flag == True:
            ab = self.vertexes[:, self.faces[1, :]] - self.vertexes[:, self.faces[0, :]]
            ac = self.vertexes[:, self.faces[2, :]] - self.vertexes[:, self.faces[0, :]]

            C = np.cross(ab.transpose(), ac.transpose())

            Ct = C.transpose()

            for b in xrange(self.face_num):
                norm = np.linalg.norm(Ct[:,b])
                nt = Ct[:,b]
                Ct[:,b] = nt / norm

            self.f_normals = Ct
            return self.f_normals
        return -1

    def calc_point_normal(self):
        """点ごとの法線"""
        if self.face_flag == True:
            self.p_normals = gni.calculate_normal(self.vertexes, self.faces)
            self.normal_flag = True
            return self.p_normals
        return -1

    def calc_center(self):
        """オブジェクトの重心"""
        c = np.sum(self.vertexes,axis=1)
        row,col = self.vertexes.shape
        c = c / float(col)
        self.center = c.reshape(3,1)
        return c.reshape(3,1)

    def calc_face_center(self):
        """"フェイスごとの重心"""
        if self.face_flag == True:
            c = self.vertexes[:, self.faces[0, :]] + \
                self.vertexes[:, self.faces[1, :]] + \
                self.vertexes[:,self.faces[2, :]]
            self.face_centers = c / 3.0
            return c/3.0

    def calc_all(self):
        self.calc_center()

        if self.face_flag == True:
            self.calc_point_normal()
            self.calc_face_center()
            self.calc_normal2()

class model3d_changeLo(model3d_plus):

    def __init__(self):
        self.local_coord = np.empty([3,4])
        """col0 center col1 x col2 y col3 z"""

    def calc_local_coord(self):

        init_axis = np.array([[0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1],
                              [1,1,1,1]])

        for a in xrange(self.face_num):
            print "f num",a

            x_axises_vec = self.vertexes[:,self.faces[:,a]] - self.face_centers[:,a].reshape(3,1)
            print x_axises_vec

            x_axis = np.empty([3, self.ver_num])
            x_axis = self.vertexes[:,self.faces[:,a]]

            print "x_axis"
            print x_axis

            z_axis = self.f_normals[:,a].reshape(3,1) + self.face_centers[:,a].reshape(3,1)
            print "z axis"
            print z_axis

            y_axis_vec = np.empty([3,3])
            for b in xrange(3):
                temp = np.cross(x_axises_vec[:,b], self.f_normals[:,a])
                y_axis_vec[:,b] = temp
                print temp


            print "y_axis"
            print y_axis_vec

            y_axis = y_axis_vec + self.face_centers[:,a].reshape(3,1)
            print "y_axis",y_axis

            solves = np.empty([4,12])

            datas = np.empty([3,3])

            for num in xrange(3):

                x = x_axis[:,num]
                data = x
                x = x/np.linalg.norm(x)

                y = y_axis[:,num]
                y = y/np.linalg.norm(y)

                z = z_axis
                z = z/np.linalg.norm(z)

                c = self.face_centers
                c = c /np.linalg.norm(c)

                zero = np.array([[1,1,1,1]])

                axis = np.c_[c,x.reshape([3,1])]
                axis = np.c_[axis,y.reshape([3,1])]
                axis = np.c_[axis,z]

                axis = np.r_[axis,zero]
                print axis
                init_inv = np.linalg.inv(init_axis.transpose())
                print init_inv

                solve = axis * init_inv
                print "solve",solve

                for num2 in xrange(self.ver_num):
                    data = self.vertexes[:,num2]

                    data = np.append(data,1)
                    test = solve.dot(data.reshape([4,1]))
                    print "test",test
                    np.savetxt("testp%03d.txt" % num2, test.reshape([1,4]), fmt='%.10f')












