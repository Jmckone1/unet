from torch.utils.data.dataset import Dataset
from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import time
import random
import torchvision.transforms.functional as TF
import torchvision
random.seed(0)
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class Test_Dataset(Dataset):
    def __init__(self, path, path_ext, size, apply_transform, **kwargs):
        self.f = []
        self.d = []
        
        self.path_ext = path_ext
        self.apply_transform = apply_transform
        
        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            counter = 0
            # each folder in extension
            for (dir_path, dir_names, file_names) in walk(path + self.path_ext[input_]):
                # gets rid of any pesky leftover .ipynb_checkpoints files
                if not dir_names == []:
                    if not dir_names[0].startswith("."):
                        
                        self.f.extend(file_names)
                        self.d.extend(dir_names)
                        counter = len(self.d)
            
            # value for extension swapping
            if input_ == 0:
                self.HGG_len = (counter) * 155
        
        # inputs to global
        self.count = len(self.d) * 155
        self.path = path
        self.size = size

    def __getitem__(self,index):

        img_data = np.empty((4,240,240,155))
        current_dir = int(np.floor(index/155))

        # assign the correct extension - HGG or LGG
        if index < self.HGG_len:
            ext = self.path_ext[0]
        else:
            ext = self.path_ext[1]

        #######################################################################
        #                          image return start                         #

        file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg_n" + '.nii.gz'
        full_path = os.path.join(self.path + ext, file_t)
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        img = img_data[:,:,:,int(index - 155*np.floor(index/155))]
        
        # interpolate image 
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))

        img = img.squeeze().numpy()
        
        #                          labels return end                          #
        #######################################################################
        
        return img
        
    def __len__(self):
        x = self.count = len(self.d) * 155
        return x # of how many examples(images?) you have