import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import importlib
try:
    ut = importlib.import_module('functions.utilities') 
except:
    print('Import Failed')
    exit()

def blobs(input,s,r,theta):
    n = int(np.ceil(3*s)*2 + 1)
    dif_kernel = ut.gaussKernel(n)
    Is = cv2.filter2D(input,-1,dif_kernel)
    (Lx,Ly) = ut.getGrad(Is)
    (Lxx,Lxy) = ut.getGrad(Lx)
    (_,Lyy) = ut.getGrad(Ly)
    
    n2 = int(np.ceil(3*r)*2 + 1)
    int_kernel = ut.gaussKernel(n2)
    Lxx = cv2.filter2D(Lxx,-1,int_kernel)
    Lyy = cv2.filter2D(Lyy,-1,int_kernel)
    Lxy = cv2.filter2D(Lxy,-1,int_kernel)


    R = Lxx * Lyy - Lxy * Lxy

    disk = ut.disk_strel(n)
    R1 = (R == cv2.dilate(R,disk,iterations = 1))
    Rmax = np.amax(R)
    R2 = R > theta * Rmax
    res = R1 & R2 
    return res

def multiscaleblobs(input,s,so,N,theta,ro):
    s_i = list(map(lambda x, y: np.power(x,y)*so,[s]*N,np.arange(N)))
    r_i = list(map(lambda x, y: np.power(x,y)*ro,[s]*N,np.arange(N)))
    blob = list(map(lambda x,y: blobs(input,x,y,theta),s_i,r_i))
    log = []
    for (i,sigma) in enumerate(s_i):
        log.append(ut.LoG(np.float32(blob[i][:][:]),sigma))
    
    trueBlobs = np.zeros(np.shape(blob))
    for i in range(N):
        if i == 0:
            trueBlobs[i] = (log[i][:][:] >= log[i+1][:][:])&(blob[i][:][:] == 1)
        elif i == N-1:
            trueBlobs[i] = (log[i][:][:] >= log[i-1][:][:])&(blob[i][:][:] == 1)
        else:
            trueBlobs[i] = (log[i][:][:] >= log[i+1][:][:])&(log[i][:][:] >= log[i-1][:][:])&(blob[i][:][:] == 1)
    
    return (trueBlobs,s_i)
