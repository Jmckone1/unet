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

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
random.seed(0)

class BraTs_Dataset(Dataset):
    def __init__(self, path, path_ext, size, apply_transform, Randomize=False, **kwargs):

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
            for (dir_path, dir_names, file_names) in walk(path + self.path_ext[input_]):
                # gets rid of any pesky leftover .ipynb_checkpoints files
                if not dir_names == []:
                    if not dir_names[0].startswith("."):
                        
                        for add_extension in range(len(dir_names)):
                            dir_names[add_extension] = self.path_ext[input_] + "/" + dir_names[add_extension]
                            
                        self.d.extend(dir_names)
                        #print(self.d)

                        counter = len(self.d)     
                        
            for directory in range(counter-c_s):
                if directory == 0:
                    if input_ == 0:
                        c_s = counter
                if input_ == 1:
                    directory = directory + c_s

                file = self.d[directory] + '/' + self.d[directory][5:] + "_" + "whimg_n" + '.nii.gz'
                full_path = path + file

                img_a = nib.load(full_path)
                img_data = img_a.get_fdata()
                
                self.index_max.extend([img_data.shape[3] + self.index_max[-1]])

                # value for extension swapping
                if input_ == 0:
                    self.HGG_len = self.index_max[-1]

        if Randomize == True:
            random.shuffle(self.d)

        # inputs to global
        self.count = self.index_max[-1] # anything with 155 in it needs to be redone to not rely on the hard coded value
        self.path = path
        self.size = size

    def __getitem__(self,index):

        print(index)
        print(self.index_max)
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

        file_t = self.d[current_dir] + '/' + self.d[current_dir][5:] + "_" + "whimg_n" + '.nii.gz'
        full_path = self.path + file_t
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        
        #print(index)
        #print(self.index_max[current_dir])
        img = img_data[:,:,:,int(index - self.index_max[current_dir])-1]
        
        # interpolate image 
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))
        
        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[current_dir] + '/' + self.d[current_dir][5:] + "_" + "whseg" + '.nii.gz'
        l_full_path = self.path + file_t
        
        l_img = nib.load(l_full_path)
        img_labels = l_img.get_fdata()
        label = img_labels[:,:,int(index - self.index_max[current_dir])-1]
        
        # interpolate label
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label,(int(label.shape[2]*self.size),int(label.shape[3]*self.size)))
        
        #                          labels return end                          #
        #######################################################################
        #                                                                     #
        
        if self.apply_transform == True:
            img,label = self.Transform(img,label)
            
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        
        #######################################################################
        #                          labels return end                          #
        #######################################################################
        
        return img,label
    
    def Transform(self, image, label):

        # 25% horizontal flip 
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 25% vertical flip 
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # rotation up to 30 degrees
        if random.random() > 0.25:
            rotation = random.randint(1,30)
            image = TF.rotate(image,rotation)
            label = TF.rotate(label,rotation)

        # 10% - 20% zoom / scaling around the center
        if random.random() > 0.25:
            size = image.shape[3]
            resize = random.randint(int(size*0.1),int(size*0.2))
            crop = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size-resize)])
            
            image = crop(image)
            image = TF.resize(image,size)

            label = crop(label)
            label = TF.resize(label,size)
        
        return image, label
        
    def __len__(self):
        x = self.index_max[-1]
        return x # of how many examples(images?) you have