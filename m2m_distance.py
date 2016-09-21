# -*- coding: utf-8 -*-

import numpy as np
import skimage.color
import model3d_kimu as m3d

"""""face to face の距離を求める
    mesh2 - mesh1 を行う
"""""

kbegin = 0
kend = 255
kmin = -5
kmax = 5

def calc_f2f_distance(mesh1,mesh2):

    f2 = mesh2.faces
    v2 = mesh2.vertexes
    f1 = mesh1.faces
    v1 = mesh1.vertexes

    row,col = f2.shape

    distList = []

    for s in xrange(col):
        dist = v2[:,f2[:,s]] - v1[:,f1[:,s]]
        distList.append(dist)

    return distList

def moveMesh(mesh1,mesh2):
    diff = mesh2.center - mesh1.center
    diffmesh = mesh1.vertexes + diff
    return diffmesh

def divideFaceDist(facedist):
    f0 = facedist[:, 0].reshape(3, 1)
    f1 = facedist[:, 1].reshape(3, 1)
    f2 = facedist[:, 2].reshape(3, 1)

    return f0,f1,f2

def modifiFaceDist(fdist0,fdist1,fdist2):
    fdist = np.c_[fdist0,fdist1,fdist2]
    return fdist

def divideDistanceList(distanceList):

    f0dists = np.empty(0)
    f1dists = np.empty(0)
    f2dists = np.empty(0)

    f0dists,f1dists,f2dists = divideFaceDist(distanceList[0])

    for fdist in distanceList[1:]:

        f0,f1,f2 = divideFaceDist(fdist)
        a = modifiFaceDist(f0,f1,f2)

        f0dists = np.c_[f0dists,f0]
        f1dists = np.c_[f1dists,f1]
        f2dists = np.c_[f2dists,f2]

    return f0dists,f1dists,f2dists

def conbineDistanceList(distanceList):

    dist = distanceList[0]

    for x in distanceList[1:]:
        dist = np.hstack((dist, x))

    return dist

def makeColorVecList(colorlist,faceNum=0,r=0,g=0,b=0):

    color = np.array([[127 for i in xrange(faceNum)],
                    [127 for i in xrange(faceNum)],
                    [127 for i in xrange(faceNum)]])
    if r != 0:
        color[0, :] = colorlist
    if g != 0:
        color[1, :] = colorlist
    if b != 0:
        color[2, :] = colorlist

    return color

def distancelist2colorlist(distanceList):
    """red:x green:y blue:z"""

    faceColors = []
    colorList = []
    facedist = divideDistanceList(distanceList)

    for x in xrange(3):
        """face is triangle from 3 points ,so x < 3 """
        distx,disty,distz = dividexyz(facedist[x])
        facecolor = []
        print "face num:",x
        distcolorx = dist2color(distx)
        distcolory = dist2color(disty)
        distcolorz = dist2color(distz)

        """0:x 1:y 2:z"""
        facecolor[0] = makeColorVecList(distcolorx, faceNum=len(distcolorx), r=1)
        facecolor[1] = makeColorVecList(distcolorx, faceNum=len(distcolory), g=1)
        facecolor[2] = makeColorVecList(distcolorx, faceNum=len(distcolorz), b=1)

        colorList[x] = facecolor

    return colorList

def dividexyz(facedist):
    distx = facedist[0,:]
    disty = facedist[1,:]
    distz = facedist[2,:]

    return distx,disty,distz

def dist2color(dist):
    b = (dist - kmin)*(kend - kbegin) / ((kmax -kmin) + kbegin)
    return b

def divide_face(mesh):
    v = mesh.vertexes
    f = mesh.faces
    ov = np.empty([3,mesh.face_num*3])
    of = np.empty([3,mesh.face_num])

    count = 0
    f_c = 0

    for face in f.transpose():

        ov[:, count] = v[:, face[0]]
        ov[:, count+1] = v[:, face[1]]
        ov[:, count + 2] = v[:, face[2]]
        of[0, f_c] = count
        of[1, f_c] = count + 1
        of[2, f_c] = count + 2

        f_c = f_c + 1
        count = count + 3

    return ov,of
