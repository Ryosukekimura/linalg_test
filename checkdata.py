# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def makeGraph(data):

    count = 0
    newdata = data[0]

    #print newdata
    for i in xrange(1,len(data)):

        print 'data',data[i]

        #newdata = np.append(newdata, data[i], axis=0)
        newdata = np.vstack([newdata, data[i]])

        count = count + 1
        if count == 0 or count == 999:
           continue

        #newdata = np.append(newdata,data[i],axis=0)
        newdata = np.vstack([newdata, data[i]])

    print 'after',newdata

    return newdata

def divideDatas(data,dir):

    if os.path.exists(dir) == False:
        os.mkdir(dir)
    count = 0
    for x in data:
        np.savetxt(dir + "/%04d.txt" % count,x.reshape(1,3),fmt='%.10f')
        count = count + 1

def main():

    incs = ['0000','0371','0457','0625','0747']

    for num in incs:
        name = 'trans_frame' + num + '_03.txt'
        data = np.loadtxt(name)
        new = makeGraph(data)
        np.savetxt('frame'+ str(num) + '_comp3_l.txt', new, fmt='%.10f')

def main2():
    name = "trans_frame0371_03.txt"
    data = np.loadtxt(name)
    divideDatas(data,'transf371')

if __name__ == "__main__":
    main2()