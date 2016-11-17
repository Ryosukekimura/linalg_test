# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def main():

    incs = ['0000','0371','0457','0625','0747']

    for num in incs:
        name = 'trans_frame' + num + '_03.txt'
        data = np.loadtxt(name)
        new = makeGraph(data)
        np.savetxt('frame'+ str(num) + '_comp3_l.txt', new, fmt='%.10f')

if __name__ == "__main__":
    main()