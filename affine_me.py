# -*- coding: utf-8 -*-
import numpy as np
import math

def conv(mat,vec):
    return mat.dot(vec)

def makeVec(x,y,z):
    p = np.array([x, y, z, 1]).reshape(4, 1)
    return p

def trans(x,y,z):
    t = np.eye(4)
    t[0,3] = x
    t[1,3] = y
    t[2,3] = z

    return t

def rotX(rad):
    rx = np.eye(4)
    rx[1,1] = np.round(math.cos(rad),decimals=8)
    rx[1,2] = np.round(-math.sin(rad),decimals=8)
    rx[2,1] = np.round(math.sin(rad),decimals=8)
    rx[2,2] = np.round(math.cos(rad),decimals=8)
    return rx

def rotY(rad):
    ry = np.eye(4)
    ry[0, 0] = np.round(math.cos(rad),decimals=8)
    ry[0, 2] = np.round(math.sin(rad),decimals=8)
    ry[2, 0] = np.round(-math.sin(rad),decimals=8)
    ry[2, 2] = np.round(math.cos(rad),decimals=8)
    return ry

def rotZ(rad):
    rz = np.eye(4)
    rz[0, 0] = np.round(math.cos(rad),decimals=8)
    rz[0, 1] = np.round(-math.sin(rad),decimals=8)
    rz[1, 0] = np.round(math.sin(rad),decimals=8)
    rz[1, 1] = np.round(math.cos(rad),decimals=8)
    rz[2, 2] = 1
    return rz

def scale(x,y,z):
    sc = np.eye(4)
    sc[0,0] = x
    sc[1,1] = y
    sc[2,2] = z
    return sc

if __name__ == "__main__":
    #test code
    rad = math.radians(90)
    p = makeVec(1,0,0)
    rz = rotZ(rad)
    t = trans(5,5,5)
    sc = scale(20,20,20)
    #print conv(rz.dot(t),p)
    #print conv(sc,p)
    #print conv(rz,p)

