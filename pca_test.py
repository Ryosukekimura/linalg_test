# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot

data = np.loadtxt('data.csv',delimiter=',')

name = ['Tanaka', 'Sato', 'Suzuki', 'Honda', 'Kawabata', 'Yoshino', 'Saito']

print data

pca = PCA(n_components= 2, whiten = False)

pca.fit(data)
print '寄与率' ,pca.explained_variance_ratio_

print '累積寄与率',np.cumsum(pca.explained_variance_ratio_)

print '主成分'
print pca.components_

x = pca.transform(data)
print 'x',x
y = pca.inverse_transform(x)
print 'y', y

"""
#データ圧縮解凍中身
data_mean = data - np.mean(data,axis=0)
data_trans = data_mean.dot(pca.components_.T)

print 'data_trans',data_trans

data_recunst = data_trans.dot(pca.components_) + np.mean(data,axis=0)

print 'data_reconst',data_recunst
"""

#pyplot.ion()
#pyplot.clf()

#データの設定
#colors = [pyplot.cm.hsv(0.1 * i,1) for i in range(len(name))]

#データのプロット
#for i in range(len(name)):
#    pyplot.scatter(x[i,0], x[i,1], c=colors[i], label=name[i])

#pyplot.legend()
#pyplot.show()
