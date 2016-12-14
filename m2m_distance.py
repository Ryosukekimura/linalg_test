# -*- coding: utf-8 -*-

import numpy as np
import math
import skimage.color
import model3d_kimu as md
import model3d_kimu_utils as mut
import get_nearplane_index as gni
import os

"""
face to face の距離を求める
mesh2 - mesh1 を行う
model3d_changeLoクラスを使用する

＊例
        m1 = m3d.model3d_changeLo()
        m2 = m3d.model3d_changeLo()

        m1.read_ply("./data/move/move%02d-shape.ply" % z)#データ読み込み
        m2.read_ply("./data/move/move%02d-vert.ply" % z)#データ読み込み

        #基本計算
        m1.calc_all()
        m2.calc_all()
        m1.calc_local_coord()

        #距離計算
        dist = md.calc_f2f_distance_in_localcoord(m1,m2)
        #distのデータ構造が複雑
        3*3*Nの配列になっている

        ＃距離を可視化する場合
        c1, c2 = md.distancelist2colorlist(dist) #c1はいらない
        nv, nf = md.divide_face(m1) #face to faceのため単純に点に色を
                                    #つけられないためfaceを分割

        #カラーリストのデータ構造が複雑
        3*3*Nの配列になっている

        mut.save_ply_file_color(name,nv,nf,faceColorxyz) #model3d_kimu_utilsの保存関数


"""

kbegin = 0
kend = 1
kmin = -0.05
kmax = 0.05

def restore_p2p_distace(distance, mesh):
    nddata = np.zeros((3, mesh.ver_num))
    newdist = np.zeros((3, mesh.ver_num))

    verCounter = np.zeros(mesh.ver_num)
    face = mesh.faces

    #print nddata

    for n in xrange(mesh.face_num):
        dt = distance[n]
        print dt[0,0]
        if dt[0,0] != dt[0,0]:
            print 'continue'
            continue

        for m in xrange(3):
            dist = distance[n]
            nddata[:,face[m,n]] += dist[:,m]
            verCounter[face[m,n]] += 1

    for n in xrange(mesh.ver_num):
        #print verCounter[n]
        if verCounter[n] == 0:
            newdist[:, n] = nddata[:, n]
        else:
            newdist[:,n] = nddata[:,n] / verCounter[n]

    #print 'temp',nddata
    #print 'rsult',newdist
    return newdist

def apply_distance_vertex(distance, invertex):
    new_ver = invertex + distance
    return new_ver

def apply_distance(distance,mesh):
    dtemp2 = restoreGlobalDisList(mesh.solves, distance) #local -> global
    newdist = restore_p2p_distace(dtemp2,mesh) #face2face dist -> p2p dist
    newver = apply_distance_vertex(newdist,mesh.vertexes) # apply dist

    remesh = md.model3d_plus()
    remesh.input_data(newver,mesh.faces)

    return remesh

def restoreGlobal(solve,distance):
    invsolve = np.linalg.inv(solve)
    init = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    dist = mut.change_local(distance,invsolve) - mut.change_local(init,invsolve)

    return dist

def  restoreGlobalDisList(solves,distance_list):
    redistList = reposit_distanceList(distance_list)
    #redistList = distance_list
    globalDisList = []

    for i in xrange(len(redistList)):

        distif = redistList[i]
        if distif[0,0] != distif[0,0]:
            box = np.array([[np.nan] for i in range(9)])
            globalDisList.append(box)
        else:
            temp = restoreGlobal(solves[i],redistList[i])
            globalDisList.append(temp)
    return globalDisList

def set_scale(max_,min_):
    global kmax,kmin
    kmax = max_
    kmin = min_

def calc_normalvec_distance(mesh1,mesh2):
    x = mesh2.vertexes
    y = mesh1.vertexes
    yf = mesh1.faces

    yn = gni.calculate_normal(y,yf)
    trans,ind,b = gni.point_to_polane_icp(x,y,yn)

    return b

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

    """
    データ構造解説
    dist = ([d0x,d1x,d2x]
            [d0y,d1y,d2y]
            [d0z,d1z,d2z])

    distList = dist0,dist1,dist2...distN

    """

    return distList

def calc_f2f_distance_in_localcoord_onlyVisible(mesh1,mesh2,visibleList):
    f2 = mesh2.faces
    v2 = mesh2.vertexes
    f1 = mesh1.faces
    v1 = mesh1.vertexes

    row, col = f2.shape

    distList = []

    for s in xrange(col):
        dist = mut.change_local(v2[:, f2[:, s]],mesh1.solves[s]) - mut.change_local(v1[:, f1[:, s]],mesh1.solves[s])
        distList.append(dist)

    """
    データ構造解説
    dist = ([d0x,d1x,d2x]
            [d0y,d1y,d2y]
            [d0z,d1z,d2z])

    distList = dist0,dist1,dist2...distN

    """

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

    #print "color",color
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

    #global kmax,kmin
    #kmax = convdist.max()
    #kmin = convdist.min()
    #print convdist.max(),convdist.min()


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
        #facecolor = []
        facecolor3 = []
        #facecolornorm = []
        #distcolorx = dist2color_fix(distx)
        #distcolory = dist2color_fix(disty)
        #distcolorz = dist2color_fix(distz)

        colorx = dist2color2(distx)
        colory = dist2color2(disty)
        colorz = dist2color2(distz)
        colorx3 = color2color_x_3(colorx)
        colory3 = color2color_x_3(colory)
        colorz3 = color2color_x_3(colorz)
        facecolor3.append(colorx3)
        facecolor3.append(colory3)
        facecolor3.append(colorz3)

        colorllist3.append(facecolor3)
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


    """
    ###データ構造解説###
    colorx = ([r0,r1,r2...rn]
             [g0,g1,g2...gn]
             [b0,b1,b2...bn]) #nはフェイスの数 np.array
    #colory,colorzも同様

    colorx3 = ([r0,r0,r0,r1,r1,r1,...rn,rn,rn]
             [g0,g0,,g0,g1,g1,g1,...gn,gn,gn]
             [b0,b0,b0,b1,b1,b1,...bn,bn,bn]) #nはフェイスの数なので大きさはn*3 np.array
    #colory,colorzも同様

    facecolor = colorx,colory,colorz #list
    facecolor3 = colorx3,colory3,colorz3 #list

    colorList = facecolor0,facecolor1,facecolor2 #list
    colorList3 = facecolor3_0,facecolor3_1,facecolor3_2 #list
    """

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

def DistanceList2DistancecVec(distanceList,faceNum):
    return np.array(distanceList).reshape(faceNum*9,1)

def change_distanceVec(distance):
    """3x3 → 9x1"""
    return distance.reshape(9,1,order = "F")

def change_distanceList(distanceList):
    """3x3 → 9x1 list用"""
    returnList = []

    for dist in distanceList:
        redist = change_distanceVec(dist)
        returnList.append(redist)

    return returnList

def reposit_distanceVec(distance):
    """9x1 → 3x3"""
    return distance.reshape(3,3,order= "F")

def reposit_distanceList(distanceList):
    """9x1 → 3x3 list用"""
    returnList = []

    for dist in distanceList:
        redist = reposit_distanceVec(dist)
        returnList.append(redist)

    return returnList

def divide_distanceList_as_f(distanceList,face_num, frameNum):
    """9x1 → 3x3 list用"""
    """with all frame"""
    """"frame 0 start"""

    returnList = []

    for i in xrange(face_num):
        temp = distanceList[i]
        temp2 = temp[:,frameNum]
        returnList.append(temp2.reshape((9,1)))

    return returnList

def conbineDistanceList(distanceList):

    dist = distanceList[0]

    for x in distanceList[1:]:
        dist = np.hstack((dist, x))

    return dist

def printColorAsPLY_xyz(mesh, distancelist, dir):
    nv, nf = divide_face(mesh)
    c, c2 = distancelist2colorlist(distancelist)

    for x in xrange(3):
        faceclors = c2[x]

        if os.path.exists(dir) == False:
            os.mkdir(dir)

        outdir = "./" + dir + "/" + "distDataface%02d/" % x
        if os.path.exists(outdir) == False:
            os.mkdir(outdir)

        xyz = ["x", "y", "z"]

        for y in xrange(3):
            faceColorxyz = faceclors[y]
            name = outdir +  xyz[y] + "dist_color_f%02d" % x + ".ply"
            print "save",name
            mut.save_ply_file_color(name, nv, nf, faceColorxyz)

def printColorAsPly(mesh,distanceList,name,if3 = True):
    col = dist2color2(distanceList)
    nv,nf = mesh.returnVF()

    if if3 == True:
        col = color2color_x_3(col)
        nv,nf = divide_face(mesh)

    mut.save_ply_file_color(name,nv,nf,col)


def averageDistance1Face(dist):

    a = dist[:, 0]
    b = dist[:, 1]
    c = dist[:, 2]

    m = (a + b + c)/3


    return m.reshape(3,1)

def averageDistnaceList(distnaceList):

    reList = []
    for x in distnaceList:
        reList.append(averageDistance1Face(x))

    return reList

def readvisible(name):
    return np.loadtxt(name,dtype=int)

def showVisibleAsColor(visibleList):
    color = np.zeros((3,len(visibleList)))

    for i in xrange(len(visibleList)):
        if visibleList[i] == 1:
            color[0,i] = 255
        else:
            color[:,i] = 127

    return color
