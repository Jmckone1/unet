from torch.utils.data.dataset import Dataset
from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt

class BraTs_Dataset(Dataset):
    def __init__(self, path, label_val, size, **kwargs):
        self.f = []
        self.d = []
        
        #self.path_ext = ["/HGG","/LGG"]
        #self.path_ext = ["/HGG_N","/LGG_N"]
        #self.path_ext = ["/HGG_N2","/LGG_N2"]
        self.path_ext = "/HGG_T"

        # each extension - HGG or LGG
        counter = 0
        # each folder in extension
        for (dir_path, dir_names, file_names) in walk(path + self.path_ext):
            self.f.extend(file_names)
            self.d.extend(dir_names)
            counter = counter + 1
        
        # inputs to global
        self.count = len(self.d) * 155
        self.path = path
        self.label_val = label_val
        self.size = size

    def __getitem__(self,index):
        #t = time.time()
        img_data = np.empty((4,240,240,155))
        img_labels = np.empty((240,240,155))
        current_dir = int(np.floor(index/155))

        #######################################################################
        #                          image return start                         #

        file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg_n3" + '.nii.gz'
        #file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg_n2" + '.nii.gz'
        #file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg_n" + '.nii.gz'
        #file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg" + '.nii.gz'
        #file_t = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whimg_o" + '.nii.gz'
        full_path = os.path.join(self.path + self.path_ext, file_t)
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        img = img_data[:,:,:,int(index - 155*np.floor(index/155))]
        
        # interpolate image 
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))
        img = img.squeeze().numpy()

        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[current_dir] + '/' + self.d[current_dir] + r"_" + "whseg" + '.nii.gz'
        l_full_path = os.path.join(self.path + self.path_ext, file_label)
        l_img = nib.load(l_full_path)
        img_labels = l_img.get_fdata()
        label = img_labels[:,:,int(index - 155*np.floor(index/155))]

        # interpolate label
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label,(int(label.shape[2]*self.size),int(label.shape[3]*self.size)))
        label = label.squeeze().numpy()

        #                          labels return end                          #
        #######################################################################

        #elapsed = time.time() - t
        #print("get_item timer =", elapsed)	
        return img,label
        
    def __len__(self):
        x = self.count = len(self.d) * 155
        return x # of how many examples(images?) you have
    
# 0 = All labels
# 1 = non-enhancing tumor core (necrotic region)
# 2 = peritumoral edema (Edema)
# 4 = GD-enhancing tumor (Active)

#dataset=BraTs_Dataset("Brats_2018 data",label_val=0)
#x,y = dataset.__getitem__(32251)

#import matplotlib.pyplot as plt
#print(y.shape)
#plt.imshow(y)
#plt.show()