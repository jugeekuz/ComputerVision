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


def harrisStephens(img,s,r,k,theta):
    #2.1.1
    n = int(np.ceil(3*s)*2 + 1)

    #Differentiation co-efficient
    dif_kernel = ut.gaussKernel(n)
    Is = cv2.filter2D(img,-1,dif_kernel)

    (dIs_dx,dIs_dy) = ut.getGrad(Is)
    (J1,J2) = ut.getGrad(dIs_dx)
    (_,J3) = ut.getGrad(dIs_dy)
    

    #Integration co-efficient
    n2 = int(np.ceil(3*r)*2 + 1)
    int_kernel = ut.gaussKernel(n2)
    J1 = cv2.filter2D(J1,-1,int_kernel)
    J2 = cv2.filter2D(J2,-1,int_kernel)
    J3 = cv2.filter2D(J3,-1,int_kernel)

    #2.1.2
    term1 = J1 + J3
    term2 = np.power((J1 - J3),2) + np.power((2*J2),2)
    term2 = np.power(term2,0.5)

    #2.1.3
    lplus = (term1 + term2)/2
    lminus = (term1 - term2)/2
    
    R = lplus * lminus - k * np.power((lplus + lminus),2)
    
    disk = ut.disk_strel(n)
    #Condition 1
    R1 = (R == cv2.dilate(R,disk,iterations = 1))
    Rmax = np.amax(R)

    #Condition2
    R2 = R > theta * Rmax
    res = R1 & R2 
    
    return res
    