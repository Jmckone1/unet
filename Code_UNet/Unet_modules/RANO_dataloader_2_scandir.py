####################################################################################################
# The Primary dataloader used for rano regression after the 04/2022 dataset normaization rework.   #
# RANO dataloader for the reduceed dataset, utilising the scandir file visulisation directive.     #
####################################################################################################

# check Test_RANO_2.py file for cleaning a improvement.

from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib
import numpy as np
import random
import torch
import sys
import os

random.seed(0)
torch.manual_seed(0)

np.set_printoptions(threshold=sys.maxsize)

image_in = "whimg_reduced"
# is this the case or does it need to be regular RANO?
# rano_in = "RANO_reduced"
rano_in = "RANO"
index_file = "/index_max_reduced.npy"

class BraTs_Dataset(Dataset):
    def __init__(self, path, path_ext, size, apply_transform, **kwargs):
        
        self.d = []
        self.index_max = []
        self.index_max.extend([0])
        
        self.path_ext = path_ext
        self.apply_transform = apply_transform
        self.HGG_len = 0
        
        c_s = 0

        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            counter = 0
            # each folder in extension
            for files in os.scandir(path + self.path_ext[input_]):
                if files.is_dir() or files.is_file():
                    if not files.name.startswith("."):
                        self.d.append(files.name)
            counter = len(self.d)
            if not os.path.exists(path + index_file):
                print("Creating index_file...")
                for directory in tqdm(range(counter-c_s)):
                    if directory == 0:
                        if input_ == 0:
                            c_s = counter
                    if input_ == 1:
                        directory = directory + c_s

                    file = self.d[directory] + '/' + self.d[directory] + "_" + image_in + '.nii.gz'
                    full_path = os.path.join(path + path_ext[input_], file)
                    img_a = nib.load(full_path)
                    img_data = img_a.get_fdata()

                    self.index_max.extend([img_data.shape[3] + self.index_max[-1]])
                
                if input_ == len(self.path_ext):
                    print("Saving index file . . . ")
                    np.save(path + index_file, self.index_max)
                    print("Index file complete")
            else:
                self.index_max = np.load(path + index_file)

                # value for extension swapping
                if input_ == 0:
                    self.HGG_len = self.index_max[-1]  
            
        # inputs to global
        self.count = self.index_max[-1]
        self.path = path
        self.size = size

    def __getitem__(self,index):

        for i in range(len(self.index_max)):
            if index >= self.index_max[i]:
                continue
            else:
                current_dir = i-1
                break
                
        # assign the correct extension - HGG or LGG
        if index < self.HGG_len:
            ext = self.path_ext[0]
        else:
            ext = self.path_ext[1]

        #######################################################################
        #                          image return start                         #

        file_t = self.d[current_dir] + '/' + self.d[current_dir] + "_" + image_in + '.nii.gz'
        full_path = os.path.join(self.path + ext, file_t)
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        img = img_data[:,:,:,int(index - self.index_max[current_dir])-1]
        
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))
        
        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[current_dir] + '/' + self.d[current_dir] + "_" + rano_in + '.npz'
        l_full_path = os.path.join(self.path + ext, file_label)
        
        l_input = np.load(l_full_path)
        label = l_input["RANO"][:,int(index - self.index_max[current_dir])-1]

        #                          labels return end                          #
        #######################################################################
        
        return img,label
        
    def __len__(self):
        x = self.index_max[-1]
        return x