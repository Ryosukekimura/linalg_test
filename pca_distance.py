# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.decomposition import PCA
import os

#データのインプット
dir = "./distancelist_new/"
files = os.listdir(dir)
datas = np.loadtxt(dir + files[0])
size = datas.size
datas.reshape(1,size)

#PCAの出力ディレクトリ作成
outdir = "./pcaData/"
if os.path.exists(outdir) == False :
    os.mkdir(outdir)

#distanceの統合
for a in files[1:]:
    file = dir + a
    print file
    temp = np.loadtxt(file)
    datas = np.vstack((datas,temp.reshape(1,size)))

datas = datas.transpose()

print "flame * points"
print "datas",datas
print datas.shape

np.savetxt("./pcaData/datas.csv", datas, fmt='%.10f', delimiter=',')

#主成分の数を増やしていって実行
comp_num = 15
for x in range(1,comp_num):
    pca = PCA(n_components=x, whiten=False)
    pca.fit(datas)
    c_ratio = pca.explained_variance_ratio_
    print '寄与率', c_ratio
    evr = np.cumsum(pca.explained_variance_ratio_)
    print '累積寄与率', evr

    print '主成分'
    print pca.components_
    print pca.components_[x-1,].reshape(1,pca.components_[x-1,].size)

    trans = pca.transform(datas)
    print 'trans', trans
    invtrans = pca.inverse_transform(trans)
    print 'invtrans',invtrans

    print "平均",pca.mean_



    # 保
    np.savetxt(outdir + "/" + "compornent%03d.csv" % x, pca.components_, fmt='%.10f', delimiter=',')
    np.savetxt(outdir + "/" + "c_ratio%03d.txt" % x, c_ratio, fmt='%.10f', delimiter=',')
    np.savetxt(outdir + "/" + "evr%03d.txt" % x, evr, fmt='%.10f', delimiter=',')
    np.savetxt(outdir + "/" + "/trans%03d.csv" % x, trans, fmt='%.10f', delimiter=',')
    np.savetxt(outdir + "/" + "invtrans%03d.csv" % x, invtrans, fmt='%.10f', delimiter=',')

    invdir = outdir + "/" + "invtrans"
    framedir = outdir + "/" + "invtrans/compornent%03d" % x

    if os.path.exists(framedir) == False:
        os.mkdir(invdir)
    if os.path.exists(invdir) == False:
        os.mkdir(framedir)

    row, col = datas.shape
    for b in xrange(col):
        filename = "invtrans_frame%03d.txt" % b

        np.savetxt(framedir +"/" + filename, invtrans[:,b], fmt='%.10f', delimiter=',')

        # 主成分１つで
    X = datas - pca.mean_
    c = pca.components_[x-1,].reshape(1,pca.components_[x-1,].size)
    X_transform = np.dot(X, c.T)

    X_transform_inv = np.dot(X_transform, c) + pca.mean_

    np.savetxt(outdir + "/" + "trans_one_comp%03d.csv" % x, X_transform, fmt='%.10f', delimiter=',')
    np.savetxt(outdir + "/" + "trans_inv_one_comp%03d.csv" % x, X_transform_inv, fmt='%.10f', delimiter=',')

    invdir = outdir + "/" + "invtrans_one"
    framedir = outdir + "/" + "invtrans_one/compornent%03d" % x

    if os.path.exists(invdir) == False:
        os.mkdir(invdir)
    if os.path.exists(framedir) == False:
        os.mkdir(framedir)

    row, col = datas.shape
    for b in range(col):
        filename = "invtrans_one_frame%03d.txt" % b
        np.savetxt(framedir + "/" + filename, invtrans[:, b], fmt='%.10f', delimiter=',')


