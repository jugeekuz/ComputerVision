import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import importlib
try:
    ut = importlib.import_module('functions.utilities') 
    hs = importlib.import_module('functions.harris_stephens') 
except:
    print('Import Failed')
    exit()


def harrisLaplacian(img,s,so,N,ro,k,theta):
    s_i = list(map(lambda x, y: np.power(x,y)*so,[s]*N,np.arange(N)))
    r_i = list(map(lambda x, y: np.power(x,y)*ro,[s]*N,np.arange(N)))
    corners = list(map(lambda x,y: hs.harrisStephens(img,x,y,k,theta),s_i,r_i))
    log = []
    for (i,sigma) in enumerate(s_i):
        log.append(ut.LoG(np.float32(corners[i][:][:]),sigma))
    
    trueCorners = np.zeros(np.shape(corners))
    for i in range(N):
        if i == 0:
            trueCorners[i] = (log[i][:][:] >= log[i+1][:][:])&(corners[i][:][:] == 1)
        elif i == N-1:
            trueCorners[i] = (log[i][:][:] >= log[i-1][:][:])&(corners[i][:][:] == 1)
        else:
            trueCorners[i] = (log[i][:][:] >= log[i+1][:][:])&(log[i][:][:] >= log[i-1][:][:])&(corners[i][:][:] == 1)
    
    return (trueCorners,s_i)
'''
img = cv2.imread('./material/edgetest_10.png')
res = ut.grayscale(img)
(corners,s) = harrisLaplacian(res,1.1,1,5,0.5)
kp_data = []
for i in range(5):
    kp_data2 = ut.makeKp_data(corners[i],s[i])
    if (len(kp_data2) != 0): 
        kp_data = kp_data + kp_data2
ax = ut.interest_points_visualization(img,kp_data)
plt.show()'''