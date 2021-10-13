#!/usr/bin/env python
import numpy as np
import math
import cv2 
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#interest_points_visualization & disk_strel were provided in assignment
#not written by me
def interest_points_visualization(I_, kp_data_, ax=None):
    '''
    Plot keypoints chosen by detectos on image.
    Args:
        I_: Image (if colored, make sure it is in RGB and not BGR).
        kp_data_: Nx3 array, as described in assignment.
        ax: Matplotlib axis to plot on (if None, a new Axes object is created).
    Returns:
        ax: Matplotlib axis where the image was plotted.
    '''
    try:
        I = np.array(I_)
        kp_data = np.array(kp_data_)
    except:
        print('Conversion to numpy arrays failed, check if the inputs (image and keypoints) are in the required format.')
        exit(2)

    try:
        assert(len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 3))
    except AssertionError as e:
        print('interest_points_visualization: Image must be either a 2D matrix or a 3D matrix with the last dimension having size equal to 3.', file=sys.stderr)
        exit(2)

    try:
        assert(len(kp_data.shape) == 2 and kp_data.shape[1] == 3)
    except AssertionError as e:
        print('interest_points_visualization: kp_data must be a 2D matrix with 3 columns.', file=sys.stderr)
        exit(2)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.imshow(I)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for i in range(len(kp_data)):
        x, y, sigma = kp_data[i]
        circ = Circle((x, y), 3*sigma, edgecolor='g', fill=False, linewidth=2)
        ax.add_patch(circ)

    return ax

def disk_strel(n):
    '''
        Return a structural element, which is a disk of radius n.
    '''
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)

def getGrad(a):
    grad = np.gradient(a)
    return (grad[0],grad[1])

def gaussKernel(n):
    kernel = cv2.getGaussianKernel(n, 0)
    kernel = kernel * kernel.transpose(1, 0)
    return kernel
    
def grayscale(rgb):
    r,g,b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    res = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return res

def findVariance(img,PSNR):
    ratio = math.pow(10,PSNR/20)
    Imax = np.amax(img)
    Imin = np.min(img)
    sn = (Imax-Imin)/ratio
    return sn

def addNoise(img,PSNR,mean):
    sn = findVariance(img,PSNR)
    row,col = img.shape
    gauss = np.random.normal(mean,sn,(row,col))
    gauss = gauss.reshape(row,col)
    res = img + gauss
    return res

def makeKp_data(input,s):
    [row,col] = np.nonzero(input)
    t = np.tile(s,len(row))
    (row,col,t) = (row.tolist(),col.tolist(),t.tolist())
    lst = []
    for i in range(len(row)):
        lst.append([col[i],row[i],t[i]])
    return lst

def LoG(input,s):
    n = int(np.ceil(3*s)*2 + 1)
    kernel = gaussKernel(n)
    input = np.array(input)
    Is = cv2.filter2D(input,-1,kernel)
    (Lx,Ly) = getGrad(Is)
    (Lxx,Lxy) = getGrad(Lx)
    (_,Lyy) = getGrad(Ly)
    return np.power(s,2)*np.abs(Lxx+Lyy)

def integralCalc(input):
    vert = np.cumsum(input,axis = 0)
    result = np.cumsum(vert,axis = 1) 
    return result