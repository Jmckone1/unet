from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import time

path_ext = ["/HGG","/LGG"]
path = "Brats_2018_data/Brats_2018_data"
output_path = ["/HGG","/LGG"]
filetype = ["t1","flair","t1ce","t2","seg"]
img_output = np.empty((4,240,240,155))
img_out = np.empty((240,240,155))
seg_output = np.zeros((240,240,155))
save = True
data_out = "whimg_n"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

f = []
d = []
for input_ in range(len(path_ext)):
    counter = 0
    for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
        f.extend(file_names)
        d.extend(dir_names)
        counter = counter + 1
    print(counter)
    if input_ == 0:
        HGG_len = counter

for num, name in enumerate(d):
    print(num)
    if num < HGG_len-1:
        ext = path_ext[0]
        out = output_path[0]
    else:
        ext = path_ext[1]
        out = output_path[1]

    t = time.time()
    for file_ in range(4):
        file_t = name + '/' + name + r"_" + filetype[file_] + '.nii.gz'
        full_path = os.path.join(path + ext, file_t)
        img = nib.load(full_path)
        
        img_input = np.zeros((240,240,155)).astype(np.float32)

        # check if slope and intercept exist in the header file
        if img.header.get_slope_inter() == (None,None):
            img.header.set_slope_inter(1,0)

        # apply the equation
        b = img.header.get_slope_inter()[0]
        m = img.header.get_slope_inter()[1]
        img_input = b * img.get_fdata() + m

        # apply the mask for normailsation, ignoring background
        # zero mean unit variance (i think)
        
        idx_nonzeros = np.nonzero(img_input)
        vol_mean = np.mean(idx_nonzeros)
        vol_std = np.std(idx_nonzeros)
        img_input = (img_input - vol_mean) / vol_std

        
        for slice_num in range(155):
            
            if file_ == 0:
                full_path = os.path.join(path + ext, name + '/' + name + "_" + filetype[4] + '.nii.gz')
                img = nib.load(full_path)
                data2 = img.get_fdata()
                
                data2[:,:,slice_num][data2[:,:,slice_num] > 0. ] = 1.
                seg_output[:,:,slice_num] = data2[:,:,slice_num]
                print(np.unique(seg_output[:,:,slice_num]))
        img_output[file_,:,:,:] = img_input
        
    seg_output_ni = nib.Nifti1Image(seg_output, np.eye(4))
    
    empty_header = nib.Nifti1Header()
    empty_header.get_data_shape()
    
    output2 = nib.Nifti1Image(img_output, img.affine, empty_header)
    new_img = output2.__class__(output2.dataobj[:], img.affine, img.header)

    # update header back to intercept = 1 and slope = 0
    new_img.header.set_slope_inter(1,0)
    new_img.header.set_data_dtype(img_input.dtype)

    print(os.path.join(path + ext, name), " : ", img_output.shape)
    if save == True:
        if not os.path.exists(os.path.join(path + out, name + '/')):
            os.makedirs(os.path.join(path + out, name + '/'))
        img_output_save = new_img
        nib.save(img_output_save, os.path.join(path + out, name + '/' + name + "_" + data_out  + '.nii.gz')) 
            
        nib.save(seg_output_ni, os.path.join(path + out, name + '/' + name + "_" + "whseg_n"  + '.nii.gz'))
    elapsed = time.time() - t
    print(elapsed)
    print(" ")