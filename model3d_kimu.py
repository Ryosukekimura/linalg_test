# -*- coding: utf-8 -*-

import numpy as np
import model3d_kimu_utils as mku
import affine_me as affine

import time
import os

"""""
model3d :
    ３次元メッシュの基本クラス
    基本的にデータの保持

model3d_plus:
    ply読み込み,法線計算,正規化,メッシュの重心計算,三角形の重心計算

model3d_changeLo :
    表面ごとのローカル座標を作成するクラス

"""""

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

    def returnVF(self):
        return self.vertexes,self.faces



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
        v, f = mku.load_ply_file(fileName)
        self.input_data(v, f)

    def write_ply(self,name):
        mku.save_ply_file(name,self.vertexes,self.faces)

    def Normalization(self,vec):

        norm = np.linalg.norm(vec)
        #print "norm",norm
        nvec = vec / norm
        #print "nvec",nvec
        return nvec

    def calc_normal2(self):
        #法線ベクトルの正規化を行っている→解除　バグあり
        if self.face_flag == True:
            ab = self.vertexes[:, self.faces[1, :]] - self.vertexes[:, self.faces[0, :]]
            ac = self.vertexes[:, self.faces[2, :]] - self.vertexes[:, self.faces[0, :]]

            C = np.cross(ab.transpose(), ac.transpose())

            Ct = C.transpose()

            #for b in xrange(self.face_num):
            #    a = self.Normalization(Ct[:,b])
            #    print "a re",a.reshape(3,1)

            self.f_normals = Ct
            #print self.f_normals
            return self.f_normals
        return -1

    def calc_point_normal(self):
        """点ごとの法線"""
        """
        if self.face_flag == True:
            self.p_normals = gni.calculate_normal(self.vertexes, self.faces)
            self.normal_flag = True
            return self.p_normals
        """
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
        print "calc normal,face center,center"
        self.calc_center()

        if self.face_flag == True:
            #self.calc_point_normal()
            self.calc_face_center()
            self.calc_normal2()



#faceごとのローカル座標を計算するクラス
"""
calc_all()
calc_local_coord()
でできる
"""

class model3d_changeLo(model3d_plus):

    def __init__(self):
        self.solves = []

        """col0 center col1 x col2 y col3 z"""

    def deform_p(self, rigid_t,vertexes):
        row, col = vertexes.shape
        dM = np.empty([3, col])

        for verNum in xrange(col):
            data = vertexes[:, verNum]
            data = np.hstack((data, 1))
            tran_p = rigid_t.dot(data.reshape([4, 1]))
            tran_p = tran_p[0:3, :]
            dM[:, verNum] = tran_p.reshape([1, 3])

        return dM.reshape([3, col])

    def calc_local_coord(self):
        """ face１つにローカル座標１つ"""

        start = time.time()

        init_axis = np.array([[0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1],
                              [1,1,1,1]])

        for fnum in xrange(self.face_num):
            #print "start calc local coord"
            face_first = self.faces[0,fnum] #faceの最初にくるやつ

            #X axis
            x_axis_vec = self.vertexes[:,face_first] - self.face_centers[:,fnum]
            x_axises_vec_no = self.Normalization(x_axis_vec)
            x_axis = x_axises_vec_no + self.face_centers[:,fnum]

            #Z axis
            z_axis = self.Normalization(self.f_normals[:,fnum]) + self.face_centers[:,fnum] #z axis point
            z_axis_n = self.Normalization(z_axis)

            # Y axis
            #print " normal",self.f_normals[:,fnum]
            y_axis_vec = np.cross(x_axises_vec_no, self.f_normals[:,fnum])
            #print "y_axis vec",y_axis_vec
            y_axis_vec_no = self.Normalization(y_axis_vec)
            y_axis = y_axis_vec_no + self.face_centers[:,fnum]

            x = x_axis
            y = y_axis
            z = z_axis
            c = self.face_centers[:,fnum]

            ones = np.array([[1,1,1,1]])

            axis = np.c_[c.reshape([3,1]),x.reshape([3,1])]
            axis = np.c_[axis,y.reshape([3,1])]
            axis = np.c_[axis,z.reshape([3,1])]
            axis = np.r_[axis,ones]
            init_inv = np.linalg.inv(init_axis)
            axis_inv = np.linalg.inv(axis)
            #print init_inv

            #solve = axis.dot(init_inv)
            solve = init_axis.dot(axis_inv)
            self.solves.append(solve)

            #print "solve",solve

        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
        return solve

    def check_local_axis(self,fnum,folderName):
        # 確認用

        outdir = "./" + folderName + "/"
        if os.path.exists(outdir) == False:
            os.mkdir(outdir)

        dm = self.deform_p(self.solves[fnum],self.vertexes)
        wri = model3d_plus()
        wri.input_data(dm,self.faces)

        wri.write_ply(outdir + "deform_test%03d.ply"%fnum)

        solve_inv = np.linalg.inv(self.solves[fnum])
        dm_inv = self.deform_p(solve_inv,dm)
        wri.input_data(dm_inv,self.faces)
        wri.write_ply(outdir + "deform_inv_test%03d.ply" % fnum)
















