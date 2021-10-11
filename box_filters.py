import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import utilities as ut
def box_filter(input,s):
    n = int(np.ceil(3*s)*2 + 1)
    integral = ut.integralCalc(input)
    Dxx, Dxy, Dyy = np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
    h = int(4*np.floor(n/6) + 1)
    l = int(2*np.floor(n/6) + 1)
    padding = (n - h)/2