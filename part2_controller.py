#Anastasios Diamantis
#03115032
#CVSP NTUA Lab Exercise Part 2

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import importlib

try:  
    hs = importlib.import_module('functions.harris_stephens')
    ut = importlib.import_module('functions.utilities')  
    hl = importlib.import_module('functions.harris_laplacian')  
    hes = importlib.import_module('functions.hessian')
except:
    print('Import Failed')
    exit()

try:
    #Read Images
    blood = cv2.imread('./material/blood_smear.jpg')
    img_b = ut.grayscale(blood)

    mars = cv2.imread('./material/mars.png')
    img_m = ut.grayscale(mars)
except:
    print('Image Read Failed')
    exit()



try:
    #2.3
    #Find and print blobs for blood_smear.jpg
    sigma = 2
    r = 2.5
    k = 0.05
    s = 1.5
    r_scale = s
    N = 4
    thetacorn = 0.005
    theta = 0.2
    blob_blood = hes.blobs(img_b,sigma,r,theta)
    kp_data_blob_blood = ut.makeKp_data(blob_blood,sigma)
    fig_bl_bl, ax_bl_bl = plt.subplots()
    ax_bl_bl = ut.interest_points_visualization(blood,kp_data_blob_blood,ax_bl_bl)
    fig_bl_bl.savefig('plots/part2/blob_blood.png')


    #Find and print blobs for mars.png
    blob_mars = hes.blobs(img_m,sigma,r,theta)
    kp_data_blob_mars = ut.makeKp_data(blob_mars,sigma)
    fig_bl_mars, ax_bl_mars = plt.subplots()
    ax_bl_bl = ut.interest_points_visualization(mars,kp_data_blob_mars,ax_bl_mars)
    fig_bl_mars.savefig('plots/part2/blob_mars.png')

    #2.4
    #Multiscale Blob Detection
    #blood_smear.jpg
    (mult_blobs_blood,s_mult_blood) = hes.multiscaleblobs(img_b,sigma,s,N,theta,r)
    kp_data_blob_blood = []
    for i in range(N):
        kp_data_temp = ut.makeKp_data(mult_blobs_blood[i],s_mult_blood[i])
        if (len(kp_data_temp) != 0): 
            kp_data_blob_blood = kp_data_blob_blood + kp_data_temp

    fig_mult_bloob_blood, ax_mult_bloob_blood = plt.subplots()
    ax_mult_bloob_blood = ut.interest_points_visualization(blood,kp_data_blob_blood,ax_mult_bloob_blood)
    fig_mult_bloob_blood.savefig('plots/part2/multiscaleblob_bloodsmear.png')

    #mars.png
    (mult_blobs_mars,s_mult_mars) = hes.multiscaleblobs(img_m,sigma,s,N,theta,r)
    kp_data_blob_mars = []
    for i in range(N):
        kp_data_temp = ut.makeKp_data(mult_blobs_mars[i],s_mult_mars[i])
        if (len(kp_data_temp) != 0): 
            kp_data_blob_mars = kp_data_blob_mars + kp_data_temp

    fig_mult_bloob_mars, ax_mult_bloob_mars = plt.subplots()
    ax_mult_bloob_mars = ut.interest_points_visualization(mars,kp_data_blob_mars,ax_mult_bloob_mars)
    fig_mult_bloob_mars.savefig('plots/part2/multiscaleblob_mars.png')
except:
    print('Unknown Error Occured')
    exit()