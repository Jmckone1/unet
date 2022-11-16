import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from os import walk
import os
import torch
import time
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Normalization():
    
    def RemoveDeprecated(path):
        if os.path.exists(path):
            os.remove(path)
            print("file " + path + " has been removed due to deprecation.")
    
    def LoadNumpy(path):
        data2 = path.get_fdata()
        normalize_numpy = data2.astype(float)
        return normalize_numpy

    def ZeroMeanNormalization(normalize_numpy, Log_output = False, mask_zeros = True, mask_numpy = None):
        
        if mask_numpy is not None:
            vol_mean = np.mean(normalize_numpy[mask_numpy > 0])
            vol_std = np.std(normalize_numpy[mask_numpy > 0])
            normalize_numpy = (normalize_numpy - vol_mean) / vol_std
            normalize_numpy[mask_numpy == 0] = 0
        elif mask_zeros:
            idx_nonzeros = np.nonzero(normalize_numpy)
            vol_mean = np.mean(normalize_numpy[idx_nonzeros])
            vol_std = np.std(normalize_numpy[idx_nonzeros])
            normalize_numpy[idx_nonzeros] = (normalize_numpy[idx_nonzeros] - vol_mean) / vol_std
        else:
            vol_mean = np.mean(normalize_numpy)
            vol_std = np.std(normalize_numpy)
            normalize_numpy = (normalize_numpy - vol_mean) / vol_std
            
        if Log_output == True:
            print("Normalization complete")

        return normalize_numpy
    
    def UnifyMask(fileName, segOutput, Log_output = False):
        img = nib.load(fileName)
        imageData = img.get_fdata()
        for slice_num in range(155):
            imageData[:,:,slice_num][imageData[:,:,slice_num] > 0. ] = 1.
            segOutput[:,:,slice_num] = imageData[:,:,slice_num]
        if Log_output == True:
            print("Unified mask")
            
    def Single_norm(path, path_ext, data_out, filetype, save = True, Log_output = False):
        #use tqdm here!!

        d = []
        for input_ in range(len(path_ext)):
            for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
                for name in range(len(dir_names)):
                    if not dir_names[name].startswith("."):
                        d.append(path_ext[input_] + "/" + dir_names[name])

        for num, name in tqdm(enumerate(d)):
            for file_ in range(len(filetype)):

                full_path = path + name + '/' + name[5:] + "_" + filetype[file_] + '.nii.gz'
                #print(full_path)

                img = nib.load(full_path)
                img_input = Normalization.LoadNumpy(img)

                # check if slope and intercept exist in the header file
                if img.header.get_slope_inter() == (None,None):
                    img.header.set_slope_inter(1,0)

                # apply the equation for slope and intercept
                b = img.header.get_slope_inter()[0]
                m = img.header.get_slope_inter()[1]
                img_input = b * img.get_fdata() + m

                #print("Normalising")
                img_output = Normalization.ZeroMeanNormalization(img_input, Log_output)

                #print("Allocating header")
                empty_header = nib.Nifti1Header()
                empty_header.get_data_shape()
                output2 = nib.Nifti1Image(img_output, img.affine, empty_header)
                new_img = output2.__class__(output2.dataobj[:], img.affine, img.header)
                new_img.header.set_slope_inter(1,0)
                new_img.header.set_data_dtype(img_input.dtype)

                if save == True:
                    #print("Saving")
                    if not os.path.exists(path + name + '/'):
                        print("Making new directory")
                        os.makedirs(path + name + '/')
                    img_output_save = new_img
                    nib.save(img_output_save, path + name + '/' + name[5:] + "_" + data_out[file_] + '.nii.gz')
                    #print("Save complete")
                    #print("")


    def RunDataset(path, path_ext, data_out, filetype, save = True, Log_output = False, remove = False):
        
        img_output = np.empty((4,240,240,155))
        img_out = np.empty((240,240,155))
        seg_output = np.zeros((240,240,155))

        d = []
        for input_ in range(len(path_ext)):
            counter = 0
            for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
                for name in range(len(dir_names)):
                    if not dir_names[name].startswith("."):
                        d.append(dir_names[name])
                        counter = counter + 1
                        
            print(counter)
            print(d)
            print(" ")
            if input_ == 0:
                HGG_len = counter

        for num, name in enumerate(d):
            if num < HGG_len:
                ext = path_ext[0]
#                 out = output_path[0]
            else:
                ext = path_ext[1]
#                 out = output_path[1]

            t = time.time()
            for file_ in range(4):
                #only do this once per file - unifes the 4 channels in the mask file into a single whole tumour segmentation
                if file_ == 0:
                    Normalization.UnifyMask(os.path.join(path + ext, name + '/' + name + "_" + filetype[4] + '.nii.gz'), seg_output, Log_output)
                    if remove == True:
                        Normalization.RemoveDeprecated(os.path.join(path + ext, name + '/' + name + '_' + 'whimg_n'  + '.nii.gz'))
                        Normalization.RemoveDeprecated(os.path.join(path + ext, name + '/' + name + '_' + 'whseg'  + '.nii.gz'))
                                                       
                file_t = name + '/' + name + "_" + filetype[file_] + '.nii.gz'
                full_path = os.path.join(path + ext, file_t)
                
                img = nib.load(full_path)
                img_input = Normalization.LoadNumpy(img)

                # check if slope and intercept exist in the header file
                if img.header.get_slope_inter() == (None,None):
                    img.header.set_slope_inter(1,0)

                # apply the equation for slope and intercept
                b = img.header.get_slope_inter()[0]
                m = img.header.get_slope_inter()[1]
                img_input = b * img.get_fdata() + m

                img_input = Normalization.ZeroMeanNormalization(img_input, Log_output)

                img_output[file_,:,:,:] = img_input

            seg_output_ni = nib.Nifti1Image(seg_output, np.eye(4))

            empty_header = nib.Nifti1Header()
            empty_header.get_data_shape()

            output2 = nib.Nifti1Image(img_output, img.affine, empty_header)
            new_img = output2.__class__(output2.dataobj[:], img.affine, img.header)

            # update header back to intercept = 1 and slope = 0
            # update the header to define datatype as float
            new_img.header.set_slope_inter(1,0)
            new_img.header.set_data_dtype(img_input.dtype)

            print(num , " : " , os.path.join(path + ext, name), " : ", img_output.shape)
            if save == True:
                if not os.path.exists(os.path.join(path + ext, name + '/')):
                    os.makedirs(os.path.join(path + ext, name + '/'))
                img_output_save = new_img
                nib.save(img_output_save, os.path.join(path + ext, name + '/' + name + "_" + data_out[0] + '.nii.gz')) 

                nib.save(seg_output_ni, os.path.join(path + ext, name + '/' + name + "_" + data_out[1] + '.nii.gz'))
            elapsed = time.time() - t
            print(elapsed)
            print(" ")
            
# def main():
    
#     path_ext = ["/HGG", "/LGG"]
#     path = "Brats_2018_data/Brats_2018_data"
#     data_out = ["whimg_norm", "whseg_norm"]
#     filetype = ["t1","flair","t1ce","t2","seg"]
    
#     Normalization.RunDataset(path, path_ext, data_out, filetype, save = True, Log_output = True, remove = True)

# if __name__ == "__main__":
#     main()