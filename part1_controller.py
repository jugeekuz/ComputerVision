#Anastasios Diamantis
#03115032
#CVSP NTUA Lab Exercise Part 1

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import importlib
try:  
    edge = importlib.import_module('functions.edge_detect')
    ut = importlib.import_module('functions.utilities')  
except:
    print('Import Failed')
    exit()
    

try:
    
    #1.1.1
    #Read image and turn to grayscale
    img = cv2.imread('./material/edgetest_10.png')
    img = ut.grayscale(img)
    plt.imsave('plots/part1/original_image.png',img,cmap = 'gray')
    #1.1.2
    #Save Image with PSNR i)10 dB ii)20dB
    #i)
    noisy_10 = ut.addNoise(img,10,0)
    plt.imsave('plots/part1/psnr10.png',noisy_10,cmap = 'gray')
    #ii)
    noisy_20 = ut.addNoise(img,20,0)
    plt.imsave('plots/part1/psnr20.png',noisy_20,cmap = 'gray')
    
    #Find original image edges
    (_,_,_,_,res_orig) = edge.edgeDetect(img,1.5,0.13,'linear')
    plt.imsave('plots/part1/res_orig.png',res_orig,cmap = 'gray')
    #1.2

    #PSNR: 20dB,s = 1.5,theta = 0.2
    #i) linear approach
    (Is,laplacian_l,bin_l,outline_l,res_l20) = edge.edgeDetect(noisy_20,1.5,0.2,'linear')
    eval_l20 = edge.evalEdges(img,res_l20,0.2)
    plt.imsave('plots/part1/res_lin20db.png',res_l20,cmap = 'gray')
    print('Evaluation for linear approach PSNR: 20dB, sn: 1.5, theta: 0.2 is: '+ str(eval_l20))

    #ii) non-linear approach
    (_,_,_,_,res_nl20) = edge.edgeDetect(noisy_20,1.5,0.2,'non-linear')
    eval_nl20 = edge.evalEdges(img,res_nl20,0.2)
    plt.imsave('plots/part1/res_nonlin20db.png',res_nl20,cmap = 'gray')
    print('Evaluation for non-linear approach PSNR: 20dB, sn: 1.5, theta: 0.2 is: '+ str(eval_nl20))

    #PSNR: 10dB,s = 3,theta = 0.2
    #linear approach
    (_,_,_,_,res_l10) = edge.edgeDetect(noisy_10,1.5,0.2,'linear')
    eval_l10 = edge.evalEdges(img,res_l10,0.2)
    plt.imsave('plots/part1/res_lin10db.png',res_l20,cmap = 'gray')
    print('Evaluation for linear approach PSNR: 10dB, sn: 1.5, theta: 0.2 is: '+ str(eval_l10))

    #non-linear
    (_,_,_,_,res_nl10) = edge.edgeDetect(noisy_10,1.5,0.2,'non-linear')
    eval_nl10 = edge.evalEdges(img,res_nl10,0.2)
    plt.imsave('plots/part1/res_nonlin10db.png',res_l20,cmap = 'gray')
    print('Evaluation for non-linear approach PSNR: 10dB, sn: 1.5, theta: 0.2 is: '+ str(eval_nl10))

    #1.4.1
    #No Noise - Urban Edges
    urban = cv2.imread('./material/urban_edges.jpg')
    img = ut.grayscale(urban) 
    #Best result : sn = 0.1 , theta = 0.2
    sn = 0.2
    theta = 0.2
    (_,_,_,_,res_nn) = edge.edgeDetect(img,sn,theta,'linear')
    eval_nn = edge.evalEdges(img,res_nn,0.2)
    plt.imsave('plots/part1/bestres_real_nonlin.png',res_nn,cmap = 'gray')
    print('Evaluation for "urban_edges" with sn :' + str(sn) + ' , theta: '+ str(theta) + ' is: ' + str(eval_nn))

    #Examples with increasing kernel size
    (_,_,_,_,res_s_1) = edge.edgeDetect(img,1,0.2,'linear')
    plt.imsave('plots/part1/res_s_1.png',res_s_1,cmap = 'gray')

    (_,_,_,_,res_s_10) = edge.edgeDetect(img,10,0.2,'linear')
    plt.imsave('plots/part1/res_s_10.png',res_s_10,cmap = 'gray')

    #Examples with increasing theta
    (_,_,_,_,res_th_0) = edge.edgeDetect(img,0.1,0,'linear')
    plt.imsave('plots/part1/res_theta_0.png',res_th_0,cmap = 'gray')

    (_,_,_,_,res_th_05) = edge.edgeDetect(img,0.1,0.5,'linear')
    plt.imsave('plots/part1/res_theta_05.png',res_th_05,cmap = 'gray')

    print('Plots Created Succesfully')
except:
    print('Error Occured While Creating Plots')
    exit()
