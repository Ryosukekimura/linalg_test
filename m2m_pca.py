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

useComp = 5

compdiv = [1,comp_max]
calcAveFlag = False

def calcPCA(data,comp_num,out,pf = False,wf = False):
    pca = PCA(n_components=comp_num, whiten=False)
    pca.fit(data)

    c_ratio = pca.explained_variance_ratio_
    evr = np.cumsum(pca.explained_variance_ratio_)

    if pf == True:
        print '寄与率', c_ratio
        print '累積寄与率', evr
        print '主成分'
        print pca.components_

    return pca.components_,c_ratio,evr

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

def test2():

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    """set output directry"""

    out = "./pcacolors/"
    if os.path.exists(out) == False:
        os.mkdir(out)

    """ read frame """

    # m1.read_ply("./data/move/move00-shape.ply")
    # m2.read_ply("./data/move/move00-vert.ply")
    print "read frame:", start_frame
    m1.read_ply("./data/reduction_data/mrshape/rdmove%04d-shape.ply"%start_frame)
    m2.read_ply("./data/reduction_data/mrvert/rdmove%04d-vert.ply"%start_frame)

    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()

    dtemp = md.calc_f2f_distance_in_localcoord(m1, m2)

    distances_ave = md.averageDistnaceList(dtemp)
    distances = md.change_distanceList(dtemp)

    for z in xrange(start_frame+1,frame):
        print "read frame:",z
        m1 = m3d.model3d_changeLo()
        m2 = m3d.model3d_changeLo()

        #m1.read_ply("./data/move/move%02d-shape.ply" % z)
        #m2.read_ply("./data/move/move%02d-vert.ply" % z)

        m1.read_ply("./data/reduction_data/mrshape/rdmove%04d-shape.ply" % z)
        m2.read_ply("./data/reduction_data/mrvert/rdmove%04d-vert.ply" % z)

        m1.calc_all()
        m2.calc_all()
        m1.calc_local_coord()

        dtemp = md.calc_f2f_distance_in_localcoord(m1,m2)
        re_dists = md.change_distanceList(dtemp)
        for i in xrange(len(distances)):
            distances[i] = np.hstack((distances[i],re_dists[i]))

    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    #m1.read_ply("./data/move/move00-shape.ply")

    c_ratio_all = np.empty(1)
    evr_all = np.empty(1)

    if calcAveFlag == True:
        distances = distances_ave

    print distances[0]

    trans_inv_list_allcomp = []

    for comp in xrange(1,comp_max):
        # calc pca
        # calc 1 face
        print "pca compornent:",comp
        trans_inv_list = []

        pca = PCA(n_components=comp, whiten=False)
        pca.fit(distances[0].T)
        c_ratio = pca.explained_variance_ratio_
        c_ratio_all_temp = c_ratio

        evr = np.cumsum(pca.explained_variance_ratio_)
        evr_all_temp = evr



        #print 'distances 0',distances[0]
        #print 'distances T', distances[0].T

        """
        print len(distances[0])
        a = distances[0]
        b = np.transpose(a[:,0])
        print "t",a[:,0].reshape((9,1))
        """
        trans_datas = pca.transform(distances[0].T)
        #trans_datas = pca.fit_transform(distances[0].T)
        trans_inv = pca.inverse_transform(trans_datas)
        trans_inv_list.append(trans_inv.T)
        #trans_inv_list.append(distances[0])
        #np.savetxt(out + "/" + "trans_frame0000_%02d.txt" % comp, trans_datas, fmt='%.10f')


        for i in xrange(1,len(distances)):
            # calc pca
            #calc more faces
            pca = PCA(n_components= comp, whiten=False)
            pca.fit(distances[i].T)
            c_ratio = pca.explained_variance_ratio_

            evr = np.cumsum(pca.explained_variance_ratio_)

            c_ratio_all_temp = np.vstack([c_ratio_all_temp, c_ratio])
            evr_all_temp = np.vstack([evr_all_temp,evr])

            # trans -> inv trans
            trans_datas = pca.transform(distances[i].T)
            #trans_datas = pca.fit_transform(distances[i].T)
            trans_inv = pca.inverse_transform(trans_datas)
            trans_inv_list.append(trans_inv.T)
            #trans_inv_list.append(distances[i])

            """"
            if i == 371 or i == 457 or i == 625 or i == 747:
                trans_datas = pca.fit_transform(distances[i].T)
                np.savetxt(out + "/" + "trans_frame%04d_%02d.txt" %(i,comp), trans_datas, fmt='%.10f')
            """
        np.savetxt(out + "/" + "c_ratiocomp%02d.txt" % comp, c_ratio_all_temp, fmt='%.10f', delimiter=',')
        np.savetxt(out + "/" + "evrcomp%02d.txt" % comp, evr_all_temp, fmt='%.10f', delimiter=',')

        c_ratio_all = c_ratio_all_temp
        #print c_ratio_all_temp.shape
        evr_all = evr_all_temp

        if comp == useComp:
            trans_inv_list_allcomp.append(trans_inv_list)

    #print trans_inv_list_allcomp[0]



    ########

    for i in xrange(start_frame,frame):
        print "restore frame:",i
        path = './data/restore/ply'
        mut.mkdir_p(path)
        shape = m3d.model3d_changeLo()
        shape.read_ply("./data/reduction_data/mrshape/rdmove%04d-shape.ply" % i)
        shape.calc_all()
        shape.calc_local_coord()

        #m2.read_ply("./data/reduction_data/mrvert/rdmove%04d-vert.ply" % i)
        inv_temp = trans_inv_list_allcomp[0]
        testlist = md.divide_distanceList_as_f(inv_temp, shape.face_num, i - start_frame)
        #print distances[0]
        #print testlist[0]
        prmesh = md.apply_distance(testlist,shape)
        prmesh.write_ply(path + '/' + 'resotre%04d.ply'%i)


        ########
        """"
        checkdist = prmesh.vertexes - m2.vertexes
        md.set_scale(0.001, -0.001)

        pathx = './data/restore/color/x'
        mut.mkdir_p(pathx)
        color = md.dist2color2(checkdist[0, :])
        prmesh.write_color_ply(pathx + '/checkdistx%04d.ply'%i, color)

        pathy = './data/restore/color/y'
        mut.mkdir_p(pathy)
        color = md.dist2color2(checkdist[1, :])
        prmesh.write_color_ply(pathy + '/checkdisty%04d.ply'%i, color)

        pathz = './data/restore/color/z'
        mut.mkdir_p(pathz)
        color = md.dist2color2(checkdist[2, :])
        prmesh.write_color_ply(pathz + '/checkdistz%04d.ply'%i, color)
        """


    """
    m = m3d.model3d_changeLo()
    m.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    nv, nf = md.divide_face(m)

    dir_r = "pcacolors/colorPly/c_ratio"
    dir_e = "pcacolors/colorPly/c_evr"
    mut.mkdir_p(dir_r)
    mut.mkdir_p(dir_e)

    #print trans_inv_list

    for i in xrange(comp_max-1):
        md.set_scale(0,1)
        col = md.dist2color2(c_ratio_all[:,i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_r + "/" + "ratio_c%03dframe0000.ply" % i, nv, nf, col3)

        col = md.dist2color2(evr_all[:,i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_e +"/" + "evr_c%03dframe0000.ply" % i, nv, nf, col3)
    """

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

