# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

import model3d_kimu as m3d
import m2m_distance as md
import os
import model3d_kimu_utils as mut

frame = 500
comp_max = 10

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

def test2():

    out = "./pcacolors/"
    if os.path.exists(out) == False:
        os.mkdir(out)

    m1 = m3d.model3d_changeLo()
    m2 = m3d.model3d_changeLo()

    # m1.read_ply("./data/move/move00-shape.ply")
    # m2.read_ply("./data/move/move00-vert.ply")

    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    m2.read_ply("./data/reduction_data/mrvert/rdmove0000-vert.ply")


    m1.calc_all()
    m2.calc_all()
    m1.calc_local_coord()

    dtemp = md.calc_f2f_distance_in_localcoord(m1, m2)
    #dtemp = md.calc_f2f_distance(m1, m2)

    distances_ave = md.averageDistnaceList(dtemp)
    distances = md.change_distanceList(dtemp)
    distances_mt = md.DistanceList2DistancecVec(distances,m1.face_num)
    #print 'change distances', distances_mt


    for z in xrange(1,frame):
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
        #dtemp = md.calc_f2f_distance(m1, m2)

        re_dists_ave = md.averageDistnaceList(dtemp)
        re_dists = md.change_distanceList(dtemp)
        distances_mt_t = md.DistanceList2DistancecVec(re_dists, m1.face_num)
        distances_mt = np.hstack((distances_mt,distances_mt_t))


        for i in xrange(len(distances)):
            distances[i] = np.hstack((distances[i],re_dists[i]))

        for i in xrange(len(distances_ave)):
            distances_ave[i] = np.hstack((distances_ave[i], re_dists_ave[i]))


    m1.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    #m1.read_ply("./data/move/move00-shape.ply")

    c_ratio_all = np.empty(1)
    evr_all = np.empty(1)

    if calcAveFlag == True:
        distances = distances_ave

    for comp in xrange(1,comp_max):

        print "pca compornent:",comp

        pca = PCA(n_components=comp, whiten=False)
        pca.fit(distances[0].T)
        c_ratio = pca.explained_variance_ratio_
        c_ratio_all_temp = c_ratio

        evr = np.cumsum(pca.explained_variance_ratio_)
        evr_all_temp = evr


        trans_datas = pca.fit_transform(distances[0].T)
        np.savetxt(out + "/" + "trans_frame0%02d.txt" % comp, trans_datas, fmt='%.10f')

        """
        trans_dist_inv = pca.inverse_transform(trans_dist)
        trans_datas.append(md.reposit_distanceVec(trans_dist_inv[:,0].reshape(9,1)))
        """

        for i in xrange(1,len(distances)):
            pca = PCA(n_components= comp, whiten=False)
            pca.fit(distances[i].T)
            c_ratio = pca.explained_variance_ratio_
            #print '寄与率', c_ratio
            evr = np.cumsum(pca.explained_variance_ratio_)
            #print '累積寄与率', evr

            #print '主成分'
            #print pca.components_


            #trans_dist = pca.fit_transform(distances[i].T)
            #trans_datas = np.vstack((trans_datas,trans_dist))

            #print "c_ratio",c_ratio
            c_ratio_all_temp = np.vstack([c_ratio_all_temp, c_ratio])
            evr_all_temp = np.vstack([evr_all_temp,evr])

        #print 'trans',trans_datas
        #np.savetxt(out + "/" + "trans%02d.txt" % comp, trans_datas, fmt='%.10f')
        np.savetxt(out + "/" + "c_ratiocomp%02d.txt" % comp, c_ratio_all_temp, fmt='%.10f', delimiter=',')
        np.savetxt(out + "/" + "evrcomp%02d.txt" % comp, evr_all_temp, fmt='%.10f', delimiter=',')

        #md.printColorAsPLY(m1,trans_datas,out + "pcaTransTestOneComp%04d"%comp)

        c_ratio_all = c_ratio_all_temp
        print c_ratio_all_temp.shape
        evr_all = evr_all_temp

    m = m3d.model3d_changeLo()
    m.read_ply("./data/reduction_data/mrshape/rdmove0000-shape.ply")
    nv, nf = md.divide_face(m)

    dir_r = "pcacolors/colorPly/c_ratio"
    dir_e = "pcacolors/colorPly/c_evr"
    mut.mkdir_p(dir_r)
    mut.mkdir_p(dir_e)

    for i in xrange(comp_max-1):
        md.set_scale(0,1)
        col = md.dist2color2(c_ratio_all[:,i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_r + "/" + "ratio_c%03dframe0000.ply" % i, nv, nf, col3)

        col = md.dist2color2(evr_all[:,i])
        col3 = md.color2color_x_3(col)

        mut.save_ply_file_color(dir_e +"/" + "evr_c%03dframe0000.ply" % i, nv, nf, col3)

def test3():

    v,f = mut.data5()
    print v
    print md.averageDistance1Face(v)

if __name__ == "__main__":

    test2()

