# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:29:02 2023

@author: Computing
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib

path_label = os.getcwd() + "/Task06_Lung/labelsRANO/lung_023_RANO.npz"
path_truth = os.getcwd() + "/Task06_Lung/imagesTr/lung_023.nii.gz"
path_truth_b = os.getcwd() + "/Task06_Lung/labelsTr/lung_023.nii.gz"
b = np.load(path_label)
x = []
for i in range(b["RANO"].shape[0]):
    if np.sum(b["RANO"][i,:]) > 0:
        print(i, b["RANO"][i,:]) 
        x.append(i)
        
print(x)
print(b["RANO"].shape)

img_a = nib.load(path_truth)
img_data = img_a.get_fdata()
print(img_data.shape)

img_b = nib.load(path_truth_b)
img_data_b = img_b.get_fdata()
print(img_data_b.shape)

# for item in x:
#     plt.imshow(img_data[:,:,item], cmap='gray')
#     D1 = np.asarray([[b["RANO"][item][1],b["RANO"][item][3]],[b["RANO"][item][0],b["RANO"][item][2]]]) 
#     D2 = np.asarray([[b["RANO"][item][5],b["RANO"][item][7]],[b["RANO"][item][4],b["RANO"][item][6]]]) 
    
#     plt.imshow(img_data_b[:,:,item])
    
#     plt.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
#     #plt.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')

#     plt.show()
    
deeplesion = [233.537, 95.0204, 234.057, 106.977, 231.169, 101.605, 236.252, 101.143]
im = plt.imread('C:/Users/Computing/Downloads/archive/minideeplesion/000001_01_01/109.png')
plt.imshow(im,"gray")

D1 = np.asarray([[deeplesion[1],deeplesion[3]],[deeplesion[0],deeplesion[2]]]) 
D2 = np.asarray([[deeplesion[5],deeplesion[7]],[deeplesion[4],deeplesion[6]]]) 
plt.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
plt.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')
plt.show()
    


