# -*- coding: utf-8 -*-

import numpy as np
import math
import skimage.color
import model3d_kimu as m3d
import model3d_kimu_utils as mut


"""""face to face の距離を求める
    mesh2 - mesh1 を行う
"""""

kbegin = 0
kend = 1
kmin = 0
kmax = 0

def calc_f2f_distance_in_localcoord(mesh1,mesh2):
    f2 = mesh2.faces
    v2 = mesh2.vertexes
    f1 = mesh1.faces
    v1 = mesh1.vertexes

    row, col = f2.shape

    distList = []

    for s in xrange(col):
        dist = mut.change_local(v2[:, f2[:, s]],mesh1.solves[s]) - mut.change_local(v1[:, f1[:, s]],mesh1.solves[s])
        distList.append(dist)

    return distList

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

    color[0, :] = colorlist

    if r != 0:
        color[0, :] = colorlist
    if g != 0:
        color[1, :] = colorlist
    if b != 0:
        color[2, :] = colorlist

    return color

def makeColorVecList2(colorlist,faceNum=0):

    color = np.array([[127 for i in xrange(faceNum)],
                    [255 for i in xrange(faceNum)],
                    [127 for i in xrange(faceNum)]])

    color = np.empty([3,faceNum])

    colorlistnp = np.array(colorlist)
    color[2, :] = colorlistnp + 127
    color[1, :] = 255 - np.absolute(colorlistnp)
    color[0, :] = 127 - colorlistnp

    return color

def vec2norm(dist):
    var,col = dist.shape
    norms = []
    for x in xrange(col):
        vec = dist[:,x]
        nr = np.linalg.norm(vec)
        norms.append(nr)
    return norms

def dist2color(dist):

    b = (dist - kmin)*(kend - kbegin) / (kmax -kmin) + kbegin

    return b

def dist2color_fix(dist):
    color = []

    for x in dist:
        # print "x",x
        if x < 0:
            a = 0 - x
        else:
            a = x

        b = (a - kmin) * (kend - kbegin) / ((kmax - kmin) + kbegin)

        if x < 0:
            b = 0 - b

        color.append(b)

    return color

def calcPseudoColor(dist,shift = 0.0):
    shift += math.pi + math.pi / 4
    calvec = np.empty(3)

    calvec[0] = int( (255 * (math.sin(1.5 * math.pi * dist + shift + math.pi) + 1.0)) / 2.0 )
    calvec[1] = int( (255 * (math.sin(1.5 * math.pi * dist + shift + math.pi / 2.0 ) + 1.0)) / 2.0 )
    calvec[2] = int( (255 * (math.sin(1.5 * math.pi * dist + shift ) + 1.0)) / 2.0 )

    #print "color", calvec.reshape([3,1])
    return calvec.reshape([3,1])

def dist2color2(dist):
    color = np.empty(0)

    b = (dist - kmin) * (kend - kbegin) / (kmax - kmin) + kbegin


    color = calcPseudoColor(b[0])
    for i in b[1:]:
        c = calcPseudoColor(i)
        color = np.hstack([color,c])

    print "color",color
    return color

def color2color_x_3(color):
    var, col = color.shape

    rcolor = np.empty([3,col*3])

    x=0
    y=0

    while x < col*3:
        #print color[:,y]
        rcolor[:,x] = color[:,y]
        rcolor[:,x+1] = color[:,y]
        rcolor[:,x+2] = color[:,y]
        y += 1
        x += 3

    return rcolor

def distancelist2colorlist(distanceList):
    """red:x green:y blue:z"""

    faceColors = []
    colorList = []
    colorllist3 = []
    facedist = divideDistanceList(distanceList)

    convdist = conbineDistanceList(distanceList)

    global kmax,kmin
    kmax = convdist.max()
    kmin = convdist.min()


    """
    if kmax < math.fabs(kmin):
        kmax = math.fabs(kmin)
    else:
        kmin = -kmax
    """

    for x in xrange(3):
        """face is triangle from 3 points ,so x < 3 """
        #print "face dist",facedist[x]
        distx,disty,distz = dividexyz(facedist[x])
        facecolor = []
        facecolor3 = []
        facecolornorm = []
        distcolorx = dist2color_fix(distx)
        distcolory = dist2color_fix(disty)
        distcolorz = dist2color_fix(distz)

        colorx = dist2color2(distx)
        colory = dist2color2(disty)
        colorz = dist2color2(distz)
        colorx3 = color2color_x_3(colorx)
        colory3 = color2color_x_3(colory)
        colorz3 = color2color_x_3(colorz)
        facecolor3.append(colorx3)
        facecolor3.append(colory3)
        facecolor3.append(colorz3)

        print "colorx",colorx

        """
        #normを計算した場合
        """
        """
        dnorms = vec2norm(facedist[x])
        kmax = max(dnorms)
        kmin = min(dnorms)
        colors = dist2color2(dnorms)
        dcpy = np.array(colors)
        dc = makeColorVecList2(dcpy,faceNum=len(dcpy))
        dc3 = color2color_x_3(dc)
        facecolornorm.append(dc3)
        facecolornorm.append(dc3)
        facecolornorm.append(dc3)
        """

        """
        facecolor.append(makeColorVecList2(distcolorx, faceNum=len(distcolorx)))
        facecolor.append(makeColorVecList2(distcolory, faceNum=len(distcolory)))
        facecolor.append(makeColorVecList2(distcolorz, faceNum=len(distcolorz)))
        colorList.append(facecolor)

        x3 = color2color_x_3(facecolor[0])
        print "x3",x3
        facecolor3.append(x3)
        y3 = color2color_x_3(facecolor[1])
        facecolor3.append(y3)
        z3 = color2color_x_3(facecolor[2])
        facecolor3.append(z3)
        """
        colorllist3.append(facecolor3)

    return colorList,colorllist3

def dividexyz(facedist):
    distx = facedist[0,:]
    disty = facedist[1,:]
    distz = facedist[2,:]

    return distx,disty,distz

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

def colorlist_conv(colorlist):

    return 0

