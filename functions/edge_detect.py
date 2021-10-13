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
    
#Finds Edges of Input Image
def edgeDetect(img,s,theta,convtype):
    n = int(np.ceil(3*s)*2 + 1)
    #1.2.1
    kernel = ut.gaussKernel(n)
    Is = cv2.filter2D(img,-1,kernel)
    #1.2.2
    if (convtype == 'linear'):
        laplacian = cv2.Laplacian(Is,cv2.CV_64F)
    elif (convtype == 'non-linear'):
        eroded = cv2.erode(Is,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1)
        dilated = cv2.dilate(Is,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1)
        laplacian = eroded + dilated - 2*Is
    #1.2.3
    binary = laplacian >= 0
    binary = binary.astype(np.uint8)
    outline = cv2.dilate(binary,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1) - cv2.erode(binary,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1)
    #1.2.4
    zerocrossings = np.zeros(binary.shape,np.uint8)
    grad = np.gradient(Is)
    grad = np.absolute(grad)
    maxgrad = np.amax(grad)
    zerocrossings = (outline == 1)&(grad > theta*maxgrad)
    res = zerocrossings[0,:,:] + zerocrossings[1,:,:]
    return (Is,laplacian,binary,outline,res)

#Evaluates Result Based on 'Real-Edges''
def evalEdges(img0,res,theta):
    eroded = cv2.erode(img0,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1)
    dilated = cv2.dilate(img0,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations = 1)
    #1.3.1
    M = dilated - eroded
    theta = theta*np.amax(M)
    T = (M > theta)
    #1.3.2
    union = (res & T)
    union = np.sum(union)
    real = np.sum(T)
    noise = np.sum(res)
    precision = union/noise
    recall = union/real
    return (precision + recall)/2




