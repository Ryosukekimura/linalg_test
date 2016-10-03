# -*- coding: utf-8 -*-

import numpy as np
import model3d_kimu as m3d
import model3d_kimu_utils as mut
import m2m_distance as md
import os
import time
import csv
from sklearn.decomposition import PCA

def calc_pca(matrix):
    pca = PCA(n_components=2)
    pca.fit(matrix)

def rearrange_dist(dist):
    d = dist[:,0]
    d = np.hstack([d,dist[:,1]])
    d = np.hstack([d, dist[:, 2]])

    return d.reshape(len(d),1)

def calc_one_frame(model1,model2):

    dist_list = md.calc_f2f_distance(model1, model2)

    frame_dist = []
    frame_dist_r = []
    for f in xrange(len(dist_list)):
        dist = mut.change_local(dist_list[f],model1.solves[f])
        frame_dist.append(dist)
        dist_r = rearrange_dist(dist)
        frame_dist_r.append(dist_r)

    return frame_dist,frame_dist_r

#####################################################

if __name__ == "__main__":

    v,f = mut.data3()
    rearrange_dist(v)


    local_dists = []
    inputdir = "./data/move"
    outdir = "./local_dist"

    files = os.listdir(inputdir)
    fnum = 1

    face_num = 2

    #出力ディレクトリ作成
    if os.path.exists(outdir) == False :
        os.mkdir(outdir)

    start = time.time()

    local_dists_r = []

    for x in xrange(face_num):
        local_dists_r.append(np.empty([9,fnum]))


    for frame in xrange(fnum):

        print "frame:",frame

        model1 = m3d.model3d_changeLo()
        model2 = m3d.model3d_changeLo()

        model1.read_ply(inputdir + "/" + "move%02d-shape.ply" % fnum)
        model2.read_ply(inputdir + "/" + "move%02d-vert.ply" % fnum)

        model1.calc_all()
        model2.calc_all()
        model1.calc_local_coord()

        dist,dist_r = calc_one_frame(model1,model2)
        local_dists.append(dist)

        for x in xrange(face_num):
            local_dists_r[x] = np.append(local_dists_r[x],dist_r[x])
            print  local_dists_r[x]

        #file = open('localdist%02d.csv' % frame,'ab')
        #cw = csv.writer(file)
        #cw.writerow(dist)


    elapsed_time = time.time() - start
    print ("finish_time:{0}".format(elapsed_time)) + "[sec]"

