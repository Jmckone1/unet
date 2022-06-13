##################################################################
# if the dataset has any slices that have no segmentation in the # 
# ground truth we remove that slice for the RANO training as it  #
# has little impact on the output but reduces the training time  #
# by half ------------------------------------------------------ #
# this is used for RANO regression training NOT segmentation     #
##################################################################

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from os import walk
import time
import os
from tqdm.auto import tqdm

# choosing the GPU number that is utilised
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def dataread(path):
    d = []
    f = []
    
    for (dir_path, dir_names, file_names) in walk(path):
        # gets rid of any pesky leftover .ipynb_checkpoints files
        if not dir_names == []:
            if not dir_names[0].startswith("."):
                f.extend(file_names)
                d.extend(dir_names)
    print(d)
    return d

name = ["HGG/","LGG/"]
input_path = "Brats_2018_data/Brats_2018_data/"

for i in range(len(name)):
    path = "Brats_2018_data/Brats_2018_data/" + name[i]

    d = dataread(path)

    output_size = len(d)
    print("Reduce Data Size: " + path)

    for x in tqdm(range(output_size)):
        
        data_Seg = nib.load(path + d[x] + "/" + d[x] + "_whseg_norm.nii.gz")
        input_1 = data_Seg.get_fdata()

        data_Plot = nib.load(path + d[x] + "/" + d[x] + "_whimg_norm.nii.gz")
        input_2 = data_Plot.get_fdata()
        
        num = 0
        for j in range(input_1.shape[2]):
            (unique, counts) = np.unique(input_1[:,:,j], return_counts=True)
            
            if len(counts) > 1:
                num = num + 1
        
        reduced_segment = np.empty((240,240,num))
        reduced_output = np.empty((4,240,240,num))
        
        tumours_found = 0
        
        for i in range(input_1.shape[2]):
            (unique, counts) = np.unique(input_1[:,:,i], return_counts=True)
            if len(counts) > 1:
                
                reduced_segment[:,:,tumours_found] = input_1[:,:,i]
                reduced_output[:,:,:,tumours_found] = input_2[:,:,:,i]
                tumours_found += 1

        red_output_save = nib.Nifti1Image(reduced_output, np.eye(4))
        nib.save(red_output_save, os.path.join(path + d[x] + "/" + d[x] +'_whimg_reduced.nii.gz'))
        
        red_segment_save = nib.Nifti1Image(reduced_segment, np.eye(4))
        nib.save(red_segment_save, os.path.join(path + d[x] + "/" + d[x] +'_whseg_reduced.nii.gz'))
