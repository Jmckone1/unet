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

class BraTs_Dataset(Dataset):
    def __init__(self, path, path_ext, size, **kwargs):
        self.f = []
        self.d = []
        
        self.path_ext = path_ext

        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            counter = 0
            # each folder in extension
            for (dir_path, dir_names, file_names) in walk(path + self.path_ext[input_]):
                self.f.extend(file_names)
                self.d.extend(dir_names)
                counter = counter + 1

            # value for extension swapping
            if input_ == 0:
              self.HGG_len = (counter-1) * 155
        
        # inputs to global
        self.count = len(self.d) * 155
        self.path = path
        self.size = size

    def __getitem__(self,index):
        img_data = np.empty((4,240,240,155))
        img_labels = np.empty((240,240,155))
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
        
 
        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whseg" + '.nii.gz'
        l_full_path = os.path.join(self.path + ext, file_label)
        l_img = nib.load(l_full_path)
        img_labels = l_img.get_fdata()
        label = img_labels[:,:,int(index - 155*np.floor(index/155))]

        # interpolate label
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label,(int(label.shape[2]*self.size),int(label.shape[3]*self.size)))
        
        img,label = self.transform(img,label)
        
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        
        #                          labels return end                          #
        #######################################################################
        
        return img,label
    
    def transform(self, image, label):

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
        x = self.count = len(self.d) * 155
        return x # of how many examples(images?) you have
