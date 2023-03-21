import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
import nibabel as nib
import numpy as np
import torchvision
import random
import torch
import sys
import os
from tqdm import tqdm
import Net_modules.Parameters_SEG as Param
from os import walk
import time

random.seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
torch.manual_seed(Param.Parameters.PRANO_Net["Global"]["Seed"])

np.set_printoptions(threshold=sys.maxsize)

class Load_Dataset(Dataset):
    def __init__(self,path,image_data, masks_data, transform=None):
        self.path = path
        self.image_folders = image_data
        self.masks_folders = masks_data
        self.transforms = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,idx):
        
        elapsed_time_fl = 0
        start = time.time()
        
        image_path = os.path.join(self.path,'imagesTr/',self.image_folders[idx] + ".nii.gz")
        img = nib.load(image_path).get_fdata()
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            mask_path = os.path.join(self.path,'BiLabelsTr/', self.masks_folders[idx] + ".npz")
                                     
            numpy_mask = np.load(mask_path)
            label = numpy_mask["RANO"][np.newaxis,:]                         
        else:
            mask_path = os.path.join(self.path,'labelsTr/',self.masks_folders[idx] + ".nii.gz")
            label = nib.load(mask_path).get_fdata()                         
            
        return (img,label)

# class Load_Dataset(Dataset):
#     def __init__(self, path, path_ext, size, apply_transform, New_index = True, **kwargs):
#         self.cap_size = 0
#         print("Init dataloader")
#         os.chdir(os.getcwd())
#         self.d = []
#         self.index_max = []
#         self.index_max.extend([0])
        
#         self.path_ext = path_ext
#         self.apply_transform = apply_transform
        
#         c_s = 0
#         # each extension - HGG or LGG
#         for input_ in range(len(self.path_ext)):
#             counter = 0
#             # each folder in extension
#             for (dir_path, dir_names, file_names) in walk(path + self.path_ext[input_]):
#                 if not dir_names == []:
#                     for file in dir_names[-self.cap_size:]:
#                         if not file[0].startswith("."):
#                             self.d.append(self.path_ext[input_] + file)
#                             counter = len(self.d)
                            
#             print("Starting new index ")
#             if New_index == True:
#                 print(self.path_ext[input_])
#                 if not os.path.exists(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"][:-9]):
#                     os.makedirs(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"][:-9])
#                 for directory in tqdm(range(counter-c_s)):
#                     if directory == 0:
#                         if input_ == 0:
#                             c_s = counter
#                     if input_ == 1:
#                         directory = directory + c_s
# #                     file = os.getcwd() +  Param.Parameters.PRANO_Net["Train_paths"]["Data_path"] + self.d[directory] + "/" + self.d[directory][4:] + "_whseg_norm.nii.gz"
#                     file = os.getcwd() +  Param.Parameters.PRANO_Net["Train_paths"]["Data_path"] + self.d[directory] + "/" + self.d[directory] + "_whseg.nii.gz"
#                     full_path = os.path.join(file)
#                     img_a = nib.load(full_path)
#                     img_data = img_a.get_fdata()
                    
#                     self.index_max.extend([img_data.shape[2] + self.index_max[-1]])
                
#                 if input_ == (len(self.path_ext)-1):
#                     print("Saving index file . . . ")
#                     np.save(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"],
#                             [self.index_max,
#                             os.getcwd() +  Param.Parameters.PRANO_Net["Train_paths"]["Data_path"] + 
#                              self.d[directory] + "/" + self.d[directory]])
#                     print("Index file complete")
#             else:
#                 self.index_max = np.load(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"]) 
            
#         self.count = self.index_max[-1]
#         self.path = path
#         self.size = size
        
#     def __getitem__(self,index):
        
#         for i in range(len(self.index_max)):
#             if index >= self.index_max[i]:
#                 continue
#             else:
#                 self.current_dir = i-1
#                 break
                
# # lung_001_reg_label.npz
# # lung_001_whimg.nii.gz
# # lung_001_whseg.nii.gz
                
#         #file_t = self.d[self.current_dir] + "/" + self.d[self.current_dir][4:] + "_flair_norm.nii.gz"
#         file_t = self.d[self.current_dir] + "/" + self.d[self.current_dir] + "_whimg.nii.gz"
        
#         full_path = self.path + file_t
#         img_a = nib.load(full_path)
#         img_data = img_a.get_fdata()
        
#         img = img_data[:,:,int(index - self.index_max[self.current_dir])-1]
        
#         img = torch.from_numpy(img).unsqueeze(0)
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
#             #file_label = self.d[self.current_dir] + "/" + self.d[self.current_dir][4:] + "_whseg_norm.nii.gz"
#             file_label = self.d[self.current_dir] + "/" + self.d[self.current_dir] + "_whseg.nii.gz"
#             l_full_path = self.path + file_label
            
#             label_a = nib.load(l_full_path)
#             label_data = label_a.get_fdata()
            
#             label = label_data[:,:,int(index - self.index_max[self.current_dir])-1]
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
#             #file_label = self.d[self.current_dir] + "/" + self.d[self.current_dir][4:] + "_RANO_2.npz"
#             file_label = self.d[self.current_dir] + "/" + self.d[self.current_dir] + "_reg_label.npz"
#             l_full_path = self.path + file_label
            
#             l_input = np.load(l_full_path)
#             label = l_input["RANO"][int(index - self.index_max[self.current_dir])-1,:]
#             label = label[np.newaxis,:]
        
#         # print(np.shape(img),np.shape(label))
#         return img,label
        
#     def __len__(self):
#         x = self.index_max[-1]
#         return x