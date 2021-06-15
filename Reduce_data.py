import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from os import walk
import time
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
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
    return d

name = ["HGG/","LGG/"]

for i in range(len(name)):
    path = "Brats_2018_data_split/Validation/" + name[i]

    d = dataread(path)

    output_size = len(d)
    print(output_size)

    for x in range(output_size):
        
        print("Data item:", x)
        
        data_Seg = nib.load(path + d[x] + "/" + d[x] + "_whseg.nii.gz")
        input_1 = data_Seg.get_fdata()

        data_Plot = nib.load(path + d[x] + "/" + d[x] + "_whimg_n.nii.gz")
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
        nib.save(red_output_save, os.path.join(path + d[x] + "/" + d[x] +'r_whimg_n.nii.gz'))
        
        red_segment_save = nib.Nifti1Image(reduced_segment, np.eye(4))
        nib.save(red_segment_save, os.path.join(path + d[x] + "/" + d[x] +'r_whseg.nii.gz'))
