# -*- coding: utf-8 -*-


"""
gitHubのeigenブランチ，mscore/python/eigen/のreg_conf.pyを使います．
学習データの準備
D
    N x J
    Nはデータ数Jは次元数
    僕のデータだと 1000 x 9 とかになる
P
    r 9,t 3をたてに並べる12パラメータ

E
    固有ベクトルを縦ベクトルとして、横に並べた行列です。
    deformationの場合、固有ベクトルが変位の次元と同じ9次元で、上位3個の固有ベクトル
    を使う場合は、9 x 3 の行列になります。

C
    Cは、係数行列です。固有ベクトルから元のデータ d を再構成するときは
    d = c_1 e_1 + c_2 e_2 … c_L e_L = c^T e
    みたいなことをすると思うのですが、この縦ベクトルcを各入力データに対応する順番
    に並べたものです。実際これを求めるのは簡単で、
    c = E^T(d - M)
    です。全部並べたものは
    C = E^T (D - M 1^T)
    （ただし、1 は全部の要素が1の縦ベクトル）で表せます

    おそらくpca.transformででてくるやつ
M
    平均値
    pca.mean_をつかえばいいかも

これらをnpz形式で保存

"""

import get_nearplane_index as gni
import model3d_kimu as m3d
import model3d_kimu_utils as mut

import numpy as np

start_frame = 30
end_frame = 100
frames = end_frame - start_frame

def makeRtVec(trans):

    trans34 = trans[0:3,:]
    rt = np.reshape(trans34.T,(12,1))
    #print rt
    return rt

def vec2mat(vec):
    mat = vec.reshape((3,4 ),order='F')
    mat44 = np.vstack((mat, np.ones(4)))
    return mat,mat44

def calcRotTrans(mesh1,mesh2):

    #mesh1 -> mesh2 rot

    mn = gni.calculate_normal(mesh2.vertexes,mesh2.faces)
    trans = gni.point_to_polane_icp(mesh1.vertexes,mesh2.vertexes,mn)
    print trans
    return trans[0]

def calcRotTranses(mNames):

    frt = np.eye(4, 4)

    rts = makeRtVec(frt)

    for i in xrange(start_frame,end_frame - 1):


        m1 = m3d.model3d_plus()
        m2 = m3d.model3d_plus()

        m1.read_ply(mNames % i)
        m2.read_ply(mNames % (i + 1))

        rt = calcRotTrans(m1,m2)

        if i == start_frame:
            numRt = rt.dot(frt)
            #print numRt
        else:
            numRt = rt.dot(numRt)
            #print numRt

        rts = np.hstack((rts, makeRtVec(numRt)))

    return rts

def setFrame(start,end):
    global start_frame,end_frame
    start_frame = start
    end_frame = end

def dataMake(dataName,datas,param = None,transdata = None,component = None,mean = None):

    np.savez(dataName,D = datas,P = param, C = transdata, E = component, M = mean)

    return 0

def dataSkip(skipFrame,params):
    #print np.delete(params,int(skipFrame),1)
    return params[:,skipFrame],np.delete(params,int(skipFrame),1)


def checkParam():
    m1 = m3d.model3d_plus()
    m1.read_ply('./data/cloth4ply/mrshape/ball0030.ply')
    rt = np.loadtxt('./pcalearn-comp3/learn/skip0040Param.txt')

    print rt
    mat,mat44 = vec2mat(rt)
    print mat44

    mr = mut.deformMesh(mat44,m1)
    mr.write_ply('defframe40,ply')

    m39 = m3d.model3d_plus()
    m40 = m3d.model3d_plus()
    m39.read_ply('./data/cloth4ply/mrshape/ball0039.ply')
    m40.read_ply('./data/cloth4ply/mrshape/ball0040.ply')
    rot = calcRotTrans(m1,m40)

    mr2 = mut.deformMesh(rot,m1)
    mr2.write_ply('defframe40-.ply')

if __name__ == "__main__":

    #m1 = m3d.model3d_plus()
    #m2 = m3d.model3d_plus()

    #m1.read_ply('./data/cloth4ply/mrshape/ball0030.ply')
    #m2.read_ply('./data/cloth4ply/mrshape/ball0040.ply')

    #calcRotTrans(m1,m2)

    checkParam()

    a = np.array([[1,2,3],[4,5,6]])








