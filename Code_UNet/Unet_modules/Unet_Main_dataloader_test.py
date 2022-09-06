import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import nibabel as nib
from os import walk
import numpy as np
import torchvision
import random
import torch
import sys
import os

from tqdm import tqdm

import Unet_modules.Parameters_seg as Param

random.seed(0)
torch.manual_seed(0)

np.set_printoptions(threshold=sys.maxsize)

class BraTs_Dataset(Dataset):
    def __init__(self, path, path_ext, size, apply_transform, **kwargs):
        
        print("Init dataloader")
        self.d = []
        self.index_max = []
        self.index_max.extend([0])
        
        self.path_ext = path_ext
        self.apply_transform = apply_transform

        c_s = 0
        self.current_dir = 0

        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            counter = 0
            # each folder in extension
            # print(path)
            for files in os.scandir(path + self.path_ext[input_]):
                if files.is_dir() or files.is_file():
                    if not files.name.startswith("."):
                        # print(self.path_ext[input_] + "/" + files.name)
                        self.d.append(self.path_ext[input_] + "/" + files.name)
            counter = len(self.d)
            # if the index file does not exist then create a new one, else load the existing one.
            # may have to implement an override in the case of a necessary deletion.
            if not os.path.exists(path + Param.sData.index_file):
                print("Creating index_file...")
                # print(path + Param.sData.index_file)
                for directory in tqdm(range(counter-c_s)):
                    if directory == 0:
                        if input_ == 0:
                            c_s = counter
                    if input_ == 1:
                        directory = directory + c_s

                    file = self.d[directory] + '/' + self.d[directory] + "_" + Param.sData.image_in + '.nii.gz'
                    full_path = os.path.join(path + path_ext[input_], file)
                    img_a = nib.load(full_path)
                    img_data = img_a.get_fdata()

                    self.index_max.extend([img_data.shape[3] + self.index_max[-1]])

                if input_ == (len(self.path_ext)-1):
                    print("Saving index file . . . ")
                    np.save(path + Param.sData.index_file, self.index_max)
                    print("Index file complete")
            else:
                self.index_max = np.load(path + Param.sData.index_file)

        # inputs to global
        self.count = self.index_max[-1]
        self.path = path
        self.size = size
        
        print("File_paths from dataloader", self.d)

    def __getitem__(self,index):
        
        self.current_dir = 0
        
        for i in range(len(self.index_max)):
            if index >= self.index_max[i]:
                continue
            else:
                self.current_dir = i-1
                break

        print(self.current_dir)
                
        #######################################################################
        #                          image return start                         #

        file_t = self.d[self.current_dir] + '/' + self.d[self.current_dir][5:] + "_" + "whimg_norm" + '.nii.gz'
        full_path = self.path + file_t
        # print(full_path)
        
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        
        img = img_data[:,:,:,int(index - self.index_max[self.current_dir])-1]
        
        # interpolate image 
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))
        
        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[self.current_dir] + '/' + self.d[self.current_dir][5:] + "_" + "whseg_norm" + '.nii.gz'
        l_full_path = self.path + file_label
        
        l_img = nib.load(l_full_path)
        img_labels = l_img.get_fdata()
        label = img_labels[:,:,int(index - self.index_max[self.current_dir])-1]
        
        # interpolate label
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label,(int(label.shape[2]*self.size),int(label.shape[3]*self.size)))
        
        #                          labels return end                          #
        #######################################################################
        
        if self.apply_transform == True:
            img,label = self.Transform(img,label)
            
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        
        return img,label,self.d[self.current_dir]
    
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