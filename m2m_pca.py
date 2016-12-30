# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

import model3d_kimu as m3d
import m2m_distance as md
import os
import model3d_kimu_utils as mut
import dataMake

start_frame = 30
frame_num = 50
frame = start_frame + frame_num
comp_max = 10 #最大次元数より+1

face_Num = 960
calcAveFlag = False

#test data
useComp = 3

skipFrame = 40
progSkipFrame = skipFrame - start_frame

#compdiv = [1,comp_max]


out = "./pcalearn-comp3/"

def readMeshes(meshName1,meshName2,AveFlag=False):
    """meshName2 - meshName2"""

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    m1.read_ply(meshName1)
    m2.read_ply(meshName2)

    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()

    dtemp = md.calc_f2f_distance_in_localcoord(m1, m2)

    if AveFlag == True:
        distances_ave = md.averageDistnaceList(dtemp)
        return distances_ave

    distances = md.change_distanceList(dtemp)
    return distances

def savedistance(dir,distance):
    path = mut.mkdir_p(dir)
    for i in xrange(face_Num):
        np.savetxt(dir + '/distface%04d.txt'%i,distance[i],fmt='%10f')

def calc_distances(visibleLists = None,VisibleFlag = False):
    """ read frame and calc local distances of all meshes"""

    print "read frame:", start_frame
    print "visible check frame ",start_frame


    m1Name = "./data/cloth4ply/mrshape/ball0000.ply"
    m2Name = "./data/cloth4ply/mrvert/cloth0000.ply"


    distances = readMeshes(m1Name, m2Name)

    if VisibleFlag is True:
        for i in xrange(len(distances)):
            if visibleLists[i, 0] == 0:
                distances[i] = None

    for z in xrange(start_frame + 1, frame):
        print "read frame:", z

        m1Name = "./data/cloth4ply/mrshape/ball%04d.ply" % z
        m2Name = "./data/cloth4ply/mrvert/cloth%04d.ply" % z

        re_dists = readMeshes(m1Name, m2Name)

        if skipFrame is not None and z == skipFrame:
            print 'skip:',z
            savedistance(out + 'frame%04d' % z,re_dists)
            continue

        if VisibleFlag is True and visibleLists is not None:
            print "visible check frame ",z
            for i in xrange(face_Num):
                check = visibleLists[i, z]

                if check == 1 and distances[i] is not None:
                    distances[i] = np.hstack((distances[i], re_dists[i]))

                elif check == 1 and distances[i] is None:
                    distances[i] = re_dists[i]

        else:
            for i in xrange(len(distances)):
                distances[i] = np.hstack((distances[i], re_dists[i]))

    return distances

def calcPCA(data,comp_num,pf = False):
    pca = PCA(n_components=comp_num, whiten=False)
    pca.fit(data)

    c_ratio = pca.explained_variance_ratio_
    evr = np.cumsum(pca.explained_variance_ratio_)
    trans = pca.transform(data)
    trans_inv = pca.inverse_transform(trans)
    mean = pca.mean_

    if pf == True:
        print '寄与率', c_ratio
        print '累積寄与率', evr
        print '主成分'
        print pca.components_

    return c_ratio,evr,pca.components_,trans,trans_inv,mean

def calcPCAallFace(comp,data,visibleList = None,VisibleFlag = False,FlagTpData = False):
    # calc pca
    # calc 1 face
    trans_inv_list = []
    trans_list = []
    n_components = []
    means = []

    if data[0] is not None:
        row, col = data[0].shape
        row = int(row)
        col = int(col)
    else:
        col = 0

    if FlagTpData is True:

        if VisibleFlag is True and data[0] is None:
            # print "visible None"
            trans_inv = np.zeros((frame, 9))
            trans_inv_list.append(trans_inv.T)
            #trans_list.append(trans_inv.T)
            c_ratio = np.zeros(comp)
            evr = np.zeros(comp)
            # print 'not'

        elif VisibleFlag is True and col == 1:
            print 'visible one '
            trans_inv = data[0].T
            trans_inv_list.append(trans_inv.T)
            c_ratio = np.ones(comp)
            evr = np.ones(comp)

        elif VisibleFlag is True and 1 < col < comp:
            print 0
            print 'rank < comp_max '
            c_ratio, evr, components, trans_datas, trans_inv, mean= calcPCA(data[0].T, col)
            trans_inv_list.append(trans_inv.T)
            trans_list.append(trans_datas)
            n_components.append(components)
            means.append(mean)

            zero = np.zeros(comp)
            one = np.ones(comp)

            for x in xrange(len(c_ratio)):
                zero[x] = c_ratio[x]
                one[x] = evr[x]

            c_ratio = zero
            evr = one

            #print c_ratio

        elif VisibleFlag is True and data[0] is not None and col >= comp:

            c_ratio, evr, components, trans_datas, trans_inv, mean= calcPCA(data[0].T, comp)
            trans_inv_list.append(trans_inv.T)
            trans_list.append(trans_datas)
            n_components.append(components)
            means.append(mean)

        else:
            c_ratio, evr, components, trans_datas, trans_inv, mean = calcPCA(data[0].T, comp)
            trans_inv_list.append(trans_inv.T)
            trans_list.append(trans_datas)
            n_components.append(components)
            means.append(mean)

    else:
        c_ratio, evr, components, trans_datas, trans_inv,mean = calcPCA(data[0], comp)
        trans_inv_list.append(trans_inv)
        trans_list.append(trans_datas)
        n_components.append(components)
        means.append(mean)

    c_ratio_all_temp = c_ratio
    evr_all_temp = evr

    for i in xrange(1, len(data)):
        # calc pca
        # calc more faces
        #print 'face',i
        if data[i] is not None:
            row, col = data[i].shape
            row = int(row)
            col = int(col)
        else:
            col = 0

        #rank = np.linalg.matrix_rank(data[i])


        if FlagTpData == True:

            if VisibleFlag is True and data[i] is None:
                # print "visible None"
                trans_inv = np.zeros((frame, 9))
                trans_inv_list.append(trans_inv.T)
                c_ratio = np.zeros(comp)
                evr = np.zeros(comp)
                # print 'not'

            elif VisibleFlag is True and col == 1:
                print 'visible one '
                trans_inv = data[i].T
                trans_inv_list.append(trans_inv.T)
                c_ratio = np.ones(comp)
                evr = np.ones(comp)

            elif VisibleFlag is True and 1 < col < comp:
                print i
                print 'rank < comp_max '
                c_ratio, evr, components, trans_datas, trans_inv, mean = calcPCA(data[i].T, col)
                trans_inv_list.append(trans_inv.T)
                trans_list.append(trans_datas)
                n_components.append(components)
                means.append(mean)
                zero = np.zeros(comp)
                one = np.ones(comp)
                for x in xrange(len(c_ratio)):
                    zero[x] = c_ratio[x]
                    one[x] = evr[x]

                c_ratio = zero
                evr = one
                print c_ratio

            elif VisibleFlag is True and data[i] is not None and col >= comp:

                c_ratio, evr, components, trans_datas, trans_inv, mean = calcPCA(data[i].T, comp)
                trans_inv_list.append(trans_inv.T)
                trans_list.append(trans_datas)
                n_components.append(components)
                means.append(mean)

            else:
                c_ratio, evr, components, trans_datas, trans_inv, mean = calcPCA(data[i].T, comp)
                trans_inv_list.append(trans_inv.T)
                trans_list.append(trans_datas)
                n_components.append(components)
                means.append(mean)


        else:
            c_ratio, evr, components, trans_datas, trans_inv, mean = calcPCA(data[i], comp)
            trans_inv_list.append(trans_inv)
            trans_list.append(trans_datas)
            n_components.append(components)
            means.append(mean)

        #print 'all',c_ratio_all_temp.shape
        #print 'i',c_ratio.shape
        #print 'data',data[i]
        #print 'all',c_ratio_all_temp
        #print 'cratio',c_ratio
        c_ratio_all_temp = np.vstack([c_ratio_all_temp, c_ratio])
        evr_all_temp = np.vstack([evr_all_temp, evr])

    return c_ratio_all_temp,evr_all_temp,n_components,trans_inv_list,trans_list,means

def calcPCAallComp():

    return 0

def saveResultPCA(outdir,applyMeshName,c_ratio_all,evr_all):

    m = m3d.model3d_changeLo()
    m.read_ply(applyMeshName)
    nv, nf = md.divide_face(m)

    dir_r = outdir + "colorPly/c_ratio"
    dir_e = outdir + "colorPly/c_evr"
    mut.mkdir_p(dir_r)
    mut.mkdir_p(dir_e)

    for i in xrange(comp_max -1):
        md.set_scale(0, 1)
        col = md.dist2color2(c_ratio_all[:, i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_r + "/" + "ratio_c%03dframe0000.ply" % i, nv, nf, col3)

        col = md.dist2color2(evr_all[:, i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_e + "/" + "evr_c%03dframe0000.ply" % i, nv, nf, col3)

        #np.savetxt(outdir + "/" + "c_ratiocomp%02d.txt" % i, c_ratio_all[:, i], fmt='%.10f', delimiter=',')
        #np.savetxt(outdir + "/" + "evrcomp%02d.txt" % i, evr_all[:, i], fmt='%.10f', delimiter=',')

def restoreMesh(meshName,invdist):
    shape = m3d.model3d_changeLo()
    shape.read_ply(meshName)
    shape.calc_all()
    shape.calc_local_coord()

    return md.apply_distance(invdist, shape)

def restoreMeshes(outdir,meshNames,invTrans,faceNum):

    for i in xrange(start_frame,frame):
        print "restore frame:",i
        print 'restore dist mat',i - start_frame
        path = outdir + '/restore/ply'
        mut.mkdir_p(path)

        name = meshNames % i

        testlist = md.divide_distanceList_as_f(invTrans, faceNum, i - start_frame)
        prmesh = restoreMesh(name,testlist)
        prmesh.write_ply(path + '/' + 'resotre%04d.ply'%i)

def restoreMesh_visible(meshName,invdist):
    shape = m3d.model3d_changeLo()
    shape.read_ply(meshName)
    shape.calc_all()
    shape.calc_local_coord()

    return md.apply_distance(invdist, shape)

def restoreMeshes_visible(outdir,meshNames,invTrans,faceNum):

    for i in xrange(start_frame,frame):
        print "restore frame:",i
        path = outdir + '/restore/ply'
        mut.mkdir_p(path)

        name = meshNames % i

        testlist = md.divide_distanceList_as_f(invTrans, faceNum, i - start_frame)

        #print testlist

        prmesh = restoreMesh_visible(name,testlist)
        prmesh.write_ply(path + '/' + 'resotre%04d.ply'%i)

def makevisibleMatrix():
    vi = md.readvisible("./data/visible/visibleFace%04d.txt" % start_frame)
    for i in xrange(start_frame + 1, frame):
        temp = md.readvisible("./data/visible/visibleFace%04d.txt" % i)
        vi = np.vstack((vi,temp))

    return vi.transpose()

def restoreVisibleDistance(data,vi,fnum):

    reDist = []
    for n in xrange(fnum):

        temp = np.where(vi[n,:] > 0)
        visible = temp[0]
        #print visible
        if visible is not None:

            box = np.array([[np.nan] * frame_num for i in range(9)])
            for x in xrange(len(visible)) :
                dtemp = data[n]

                a =visible[x]

                box[:, a] = dtemp[:, x]

            reDist.append(box)
        else:
            box = np.array([[np.nan] * frame_num for i in range(9)])
            reDist.append(box)

    return reDist
######################### test code #############################################

def dataMakesAllface(datas,trans,components, means,params,skip = None):


    rts = params

    old = mut.mkdir_p(out + 'learn/')
    if skip is not None:
        skipp,rts = dataMake.dataSkip(skip, params)
        print skipp
        np.savetxt(old + '/skip%04dParam.txt'%(skip + start_frame),skipp,delimiter=',')

    for i in xrange(face_Num):
        #print 'face:',i
        name = old + '/testf%04d.npz' % i
        dataMake.dataMake(dataName= name, datas=datas[i].T, param=rts, transdata=trans[i].T, component=components[i].T, mean=means[i])


def test2():

    m1 = m3d.model3d_changeLo()
    m1.read_ply("./data/cloth4ply/mrshape/ball0000.ply")

    """set output directry"""
    if os.path.exists(out) == False:
        os.mkdir(out)

    distances = calc_distances()
    a = distances[0]
    print a.shape
    c_ratio_all = np.empty(1)
    evr_all = np.empty(1)

    #print distances[0]

    trans_inv_list_allcomp = []
    trans_list_allcomp = []
    n_components_allcomp = []
    mean_allcomp = []

    for comp in xrange(1,comp_max):
        # calc pca use component 1 -> component max
        print "pca compornent:",comp

        c_ratio_all_temp,evr_all_temp,components,trans_inv_list,trans_list, means = calcPCAallFace(comp, distances, FlagTpData=True)

        c_ratio_all = c_ratio_all_temp
        evr_all = evr_all_temp

        if comp == useComp:
            trans_inv_list_allcomp.append(trans_inv_list)
            mean_allcomp.append(means)
            trans_list_allcomp.append(trans_list)
            n_components_allcomp.append(components)

        else:
            trans_inv_list_allcomp.append(None)
            mean_allcomp.append(None)
            trans_list_allcomp.append(None)
            n_components_allcomp.append(None)

        np.savetxt(out + "/" + "c_ratiocomp%02d.txt" % comp, c_ratio_all_temp, fmt='%.20f', delimiter=',')
        np.savetxt(out + "/" + "evrcomp%02d.txt" % comp, evr_all_temp, fmt='%.20f', delimiter=',')

    m = mean_allcomp[useComp -1]
    t = trans_list_allcomp[useComp -1]
    c = n_components_allcomp[useComp -1]

    dataMake.setFrame(start_frame, frame)
    rts = dataMake.calcRotTranses("./data/cloth4ply/mrshape/ball%04d.ply")


    dataMakesAllface(distances, t, c, m,rts, progSkipFrame)


    #restoreMeshes(out,"./data/cloth4ply/mrshape/ball%04d.ply",trans_inv_list_allcomp[useComp - 1],m1.face_num)
    saveResultPCA(out,"./data/cloth4ply/mrshape/ball0000.ply",c_ratio_all,evr_all)



def test3():

    m1 = m3d.model3d_changeLo()
    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")

    """set output directry"""
    if os.path.exists(out) == False:
        os.mkdir(out)

    visibleList = makevisibleMatrix()

    distances = calc_distances(visibleList,VisibleFlag=True)



    #for i in xrange(len(distances)):
    #    a = distances[i]
    #    #print 'a',a

    c_ratio_all = np.empty(1)
    evr_all = np.empty(1)
    trans_inv_list_allcomp = []

    for comp in xrange(1, comp_max):
        # calc pca use component 1 -> component max
        print "pca compornent:", comp

        c_ratio_all_temp, evr_all_temp, trans_inv_list = calcPCAallFace(comp, distances,visibleList,VisibleFlag=True, FlagTpData=True)

        c_ratio_all = c_ratio_all_temp
        evr_all = evr_all_temp

        if comp == useComp:
            trans_inv_list_allcomp.append(trans_inv_list)

        np.savetxt(out + "/" + "c_ratiocomp%02d.txt" % comp, c_ratio_all_temp, fmt='%.10f', delimiter=',')
        np.savetxt(out + "/" + "evrcomp%02d.txt" % comp, evr_all_temp, fmt='%.10f', delimiter=',')

    redist = restoreVisibleDistance(trans_inv_list_allcomp[0], visibleList, 1000)

    restoreMeshes_visible(out, "./data/reduction_data/mrshape/rdmove%04d-shape.ply",redist , 1000)
    saveResultPCA(out, "./data/reduction_data/mrshape/rdmove0000-shape.ply", c_ratio_all, evr_all)

    return 0

def test4():

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    m1.read_ply("./data/cloth4ply/mrshape/ball%04d.ply"%50)
    m2.read_ply("./data/cloth4ply/mrvert/cloth%04d.ply"%50)

    print 'fnum',m1.face_num

    v,f = mut.data5()
    #m1.input_data(v,f)
    v,f = mut.data6()
    #m2.input_data(v,f)


    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()

    #m1.write_ply('m1.ply')
    #m2.write_ply('m2.ply')

    #m1.check_local_axis_2(0,'chccklo',m2)

    dtemp = md.calc_f2f_distance_in_localcoord(m1, m2)
    re_dists = md.change_distanceList(dtemp)
    remesh = md.apply_distance(re_dists,m1)

    print "local dist",dtemp

    print "exchange",re_dists

    checkver = remesh.vertexes - m2.vertexes

    np.savetxt("checkdistance0000.txt", checkver, delimiter=',')

    md.set_scale(0.00000000000000001, -0.00000000000000001)
    color = md.dist2color2(checkver[0,:])
    print color
    remesh.write_color_ply('checkdist.ply', color)
    remesh.write_color_ply('red.ply',mut.makeRed(remesh.ver_num))

    std = np.std(checkver)
    print 'std',std

    er = checkver / m2.vertexes
    er = np.absolute(er)
    print er

    print er.mean()
    print er.max()
    std = np.std(er)
    print 'std', std

    np.savetxt("error0300.txt", er, delimiter=',')

    """"
    print 'local',dtemp
    print '**********'
    print 'restore',dtemp2
    print '**********'
    print 'global',dtemp3
    """

def test5():

    red = mut.makeRed(1000)
    path = './data/restore-pass-comp9'
    mut.mkdir_p(path)
    for i in xrange(1000):
        m1 = m3d.model3d_plus()
        m1.read_ply("./pcacolors-comp9/restore/ply/resotre%04d.ply"%i)
        m1.write_color_ply(path + '/restore%04d.ply'%i,red)

def test6():

    vi = md.readvisible("./data/visible/visibleFace0319.txt")
    m = m3d.model3d_changeLo()
    m.read_ply("./data/reduction_data/mrvert/rdmove0319-vert.ply")
    v,f = md.divide_face(m)
    c = md.showVisibleAsColor(vi)
    c2 = md.color2color_x_3(c)
    print v.shape
    mut.save_ply_file_color("visibleCheck.ply",v,f,c2)

    return 0

def test7():
    vi = makevisibleMatrix()
    np.savetxt("visual.txt",vi,fmt='%d', delimiter=',')

    dist = calc_distances(vi, VisibleFlag=True)
    redist = restoreVisibleDistance(dist,vi,1000)

    restoreMeshes_visible("./", "./data/reduction_data/mrshape/rdmove%04d-shape.ply", redist, 1000)
    """
    for i in xrange(len(dist)):
        if dist[i] is not None:
            print dist[i].shape
        else:
            print 'None'
    """

def test8():
    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    m2.read_ply("./restore/ply/resotre0000.ply")

    m1.calc_all()
    m1.calc_local_coord()

    dist = md.calc_normalvec_distance(m1,m2)

    #dist = md.calc_f2f_distance_in_localcoord(m1,m2)
    md.printColorAsPly(m2,dist,'restoreCheck0000.ply',if3 = False)


if __name__ == "__main__":

    test2()

