# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

import model3d_kimu as m3d
import m2m_distance as md
import os
import model3d_kimu_utils as mut

start_frame = 0
frame_num = 1000
frame = start_frame + frame_num
comp_max = 10 #最大次元数より+1

useComp = 9

compdiv = [1,comp_max]
calcAveFlag = False

out = "./pcacolors/"

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

def calc_distances():
    """ read frame and calc local distances of all meshes"""

    print "read frame:", start_frame
    m1Name = "./data/reduction_data/mrshape/rdmove%04d-shape.ply" % start_frame
    m2Name = "./data/reduction_data/mrvert/rdmove%04d-vert.ply" % start_frame

    distances = readMeshes(m1Name, m2Name)

    for z in xrange(start_frame + 1, frame):
        print "read frame:", z

        m1Name = "./data/reduction_data/mrshape/rdmove%04d-shape.ply" % z
        m2Name = "./data/reduction_data/mrvert/rdmove%04d-vert.ply" % z

        re_dists = readMeshes(m1Name, m2Name)

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

    if pf == True:
        print '寄与率', c_ratio
        print '累積寄与率', evr
        print '主成分'
        print pca.components_

    return c_ratio,evr,pca.components_,trans,trans_inv

def calcPCAallFace(comp,data,FlagTpData = False):
    # calc pca
    # calc 1 face
    trans_inv_list = []

    if FlagTpData == True:
        c_ratio, evr, components, trans_datas, trans_inv = calcPCA(data[0].T, comp)
        trans_inv_list.append(trans_inv.T)
    else:
        c_ratio, evr, components, trans_datas, trans_inv = calcPCA(data[0], comp)
        trans_inv_list.append(trans_inv)

    c_ratio_all_temp = c_ratio
    evr_all_temp = evr

    for i in xrange(1, len(data)):
        # calc pca
        # calc more faces
        if FlagTpData == True:
            c_ratio, evr, components, trans_datas, trans_inv = calcPCA(data[i].T, comp)
            trans_inv_list.append(trans_inv.T)
        else:
            c_ratio, evr, components, trans_datas, trans_inv = calcPCA(data[i], comp)
            trans_inv_list.append(trans_inv)

        c_ratio_all_temp = np.vstack([c_ratio_all_temp, c_ratio])
        evr_all_temp = np.vstack([evr_all_temp, evr])

    return c_ratio_all_temp,evr_all_temp,trans_inv_list

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
        path = outdir + '/restore/ply'
        mut.mkdir_p(path)

        name = meshNames % i

        testlist = md.divide_distanceList_as_f(invTrans, faceNum, i - start_frame)
        prmesh = restoreMesh(name,testlist)
        prmesh.write_ply(path + '/' + 'resotre%04d.ply'%i)

def test2():

    m1 = m3d.model3d_changeLo()
    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")

    """set output directry"""
    if os.path.exists(out) == False:
        os.mkdir(out)

    distances = calc_distances()

    c_ratio_all = np.empty(1)
    evr_all = np.empty(1)

    #print distances[0]

    trans_inv_list_allcomp = []

    for comp in xrange(1,comp_max):
        # calc pca use component 1 -> component max
        print "pca compornent:",comp

        c_ratio_all_temp,evr_all_temp,trans_inv_list = calcPCAallFace(comp,distances,FlagTpData=True)

        c_ratio_all = c_ratio_all_temp
        evr_all = evr_all_temp

        if comp == useComp:
            trans_inv_list_allcomp.append(trans_inv_list)

        np.savetxt(out + "/" + "c_ratiocomp%02d.txt" % comp, c_ratio_all_temp, fmt='%.10f', delimiter=',')
        np.savetxt(out + "/" + "evrcomp%02d.txt" % comp, evr_all_temp, fmt='%.10f', delimiter=',')

    restoreMeshes(out,"./data/reduction_data/mrshape/rdmove%04d-shape.ply",trans_inv_list_allcomp[0],1000)
    saveResultPCA(out,"./data/reduction_data/mrshape/rdmove0000-shape.ply",c_ratio_all,evr_all)

def test3():

    v,f = mut.data5()
    print v
    print md.averageDistance1Face(v)

def test4():

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    m1.read_ply("./data/reduction_data/mrshape/rdmove0100-shape.ply")
    m2.read_ply("./data/reduction_data/mrvert/rdmove0100-vert.ply")

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
    path = './data/restoreColor'
    mut.mkdir_p(path)
    for i in xrange(1000):
        m1 = m3d.model3d_plus()
        m1.read_ply("./data/restore/ply/resotre%04d.ply"%i)
        m1.write_color_ply(path + '/restore%04d.ply'%i,red)

def test6():

    return 0


if __name__ == "__main__":

    test2()

