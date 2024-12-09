import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
#import torch.nn.functional as F
import nibabel as nib
import numpy as np
import torchvision
import random
import torch
import sys
import os

from tqdm import tqdm

import Net_modules.Parameters_PRANO as Param
from os import walk

random.seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
torch.manual_seed(Param.Parameters.PRANO_Net["Global"]["Seed"])

np.set_printoptions(threshold=sys.maxsize)

cap_size = 0

class Load_Dataset(Dataset):
    def __init__(self, path, path_ext, size, apply_transform, New_index = True, **kwargs):
        
        print("Init dataloader")
        os.chdir(os.getcwd())
        self.d = []
        self.index_max = []
        self.index_max.extend([0])
        
        self.path_ext = path_ext
        self.apply_transform = apply_transform
        
        c_s = 0
        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            counter = 0
            # each folder in extension
            # print(path + self.path_ext[input_])
            for (dir_path, dir_names, file_names) in walk(path + self.path_ext[input_]):
                # gets rid of any pesky leftover .ipynb_checkpoints files
                print(file_names)
                if not file_names == []:
                    # caps the amount of data items in the set loaded
                    for file in file_names[-cap_size:]:
                        if not file[0].startswith("."):
                           # print(file)
                            self.d.append(file)
                            counter = len(self.d)
                            
            print(self.d)
            if New_index == True:
                print("Starting new index")
                # print(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"][:-9])
                if not os.path.exists(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"][:-9]):
                    os.makedirs(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"][:-9])
                # print("Creating index_file...")
                # print(Param.Parameters.PRANO_Net["Train_paths"]["Data_path"])
                for directory in tqdm(range(counter-c_s)):
                    if directory == 0:
                        if input_ == 0:
                            c_s = counter
                    if input_ == 1:
                        directory = directory + c_s
                    
                    file = Param.Parameters.PRANO_Net["Train_paths"]["Data_path"] + self.d[directory]
                    full_path = os.path.join(os.getcwd() + file)
                    img_a = nib.load(full_path)
                    img_data = img_a.get_fdata()
                    
                    self.index_max.extend([img_data.shape[2] + self.index_max[-1]])
                    # print(img_data.shape[3] + self.index_max[-1])
                
                if input_ == (len(self.path_ext)-1):
                    print("Saving index file . . . ")
                    np.save(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"],self.index_max)
                    print("Index file complete")
            else:
                self.index_max = np.load(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"]) 
            
        # inputs to global
        self.count = self.index_max[-1] # anything with 155 in it needs to be redone to not rely on the hard coded value
        self.path = path
        self.size = size
        
        print("File_paths from dataloader", self.d)

    def __getitem__(self,index):

        for i in range(len(self.index_max)):
            if index >= self.index_max[i]:
                continue
            else:
                self.current_dir = i-1
                break

        file_t = self.d[self.current_dir][:-7] + ".nii.gz"
        full_path = self.path + file_t
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        
        img = img_data[:,:,int(index - self.index_max[self.current_dir])-1]
        
        img = torch.from_numpy(img).unsqueeze(0)
        
        file_label = "labelsRANO/" + self.d[self.current_dir][:-7] + "_RANO.npz"
        # print(file_label)
        l_full_path = self.path[:-9] + file_label
        # print(l_full_path)
        
        l_input = np.load(l_full_path)
        # print(l_input["RANO"].shape)
        label = l_input["RANO"][int(index - self.index_max[self.current_dir])-1,:]

        img = img.squeeze().numpy()

        return img,label
        
    def __len__(self):
        x = self.index_max[-1]
        return x