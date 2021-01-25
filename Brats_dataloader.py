import logging
from torch.utils.data.dataset import Dataset
from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as transforms

class BraTs_Dataset(Dataset):
    def __init__(self, path,label_val, **kwargs):
        self.f = []
        self.d = []
        for (dir_path, dir_names, file_names) in walk(path):
            self.f.extend(file_names)
            self.d.extend(dir_names)

        self.count = len(self.d) * 155
        self.path = path
        self.filetype = ["t1","flair","t1ce","t2","seg","whseg"]
        self.label_val = label_val

    def __getitem__(self,index):
        # stuff
        bin_size = 240 // 120
        img_data = np.empty((4,240,240,155))
        img_labels = np.empty((240,240,155))
        current_dir = int(np.floor(index/155))

        # if input is at 0 (default) use the full segmentation
        # otherwise use the region
        # need to add a break in case the number is
        if self.label_val == 0:
          label_file = self.filetype[5]
        else:
          label_file = self.filetype[4]

        for i in range(4):
            file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + self.filetype[i] + '.nii.gz'
            full_path = os.path.join(self.path, file_t)
            img = nib.load(full_path)
            img_data[i,:,:,:] = (img.get_fdata())[:,:,:]
        img = img_data[:,:,:,int(index - 155*np.floor(index/155))]
        #img = img.reshape(240,240,4)
        
#        img = img.reshape((4, 60, bin_size,
#                              60, bin_size)).max(4).max(2)
        
        file_label = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + label_file + '.nii.gz'
        l_full_path = os.path.join(self.path, file_label)
        l_img = nib.load(l_full_path)
        img_labels[:,:,:] = (l_img.get_fdata())[:,:,:]
        
        if self.label_val != 0:
          img_labels = (img_labels == self.label_val).astype(float)

#        label = label.reshape((1,60, bin_size,60, bin_size)).max(4).max(2)
        label = img_labels[:,:,int(index - 155*np.floor(index/155))]
        return img,label
        
    def __len__(self):
        x = self.count = len(self.d) * 155
        return x # of how many examples(images?) you have
    
# 0 = All labels
# 1 = non-enhancing tumor core (necrotic region)
# 2 = peritumoral edema (Edema)
# 4 = GD-enhancing tumor (Active)
