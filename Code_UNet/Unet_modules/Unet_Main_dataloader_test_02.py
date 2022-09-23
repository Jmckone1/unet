# code includes an experiment to test the output of the total number of files and whether the slice allocation / detection algortihm works in this case by performing modulus on the index file values. this is the file for the test values.
import torchvision
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
        self.path_ext = path_ext
        self.apply_transform = apply_transform

        c_s = 0
        self.current_dir = 0

        # each extension - HGG or LGG
        for input_ in range(len(self.path_ext)):
            for files in os.scandir(path + self.path_ext[input_]):
                print("#############################",files)
                if files.is_dir() or files.is_file():
                    if not files.name.startswith("."):
                        self.d.append(self.path_ext[input_] + "/" + files.name)
                        
        self.count = len(self.d)*155
        self.path = path
        self.size = size
        # finding the length of the dataset in total (per number of volumes), finding the recorded max index number (per number of slices) then performing the equation to infer one from the other to check that they line up.
        print("length of the dataset files", len(self.d))

        print("Final value", int(self.count))
        print("Final value of the index max", int(self.count / 155))
#         print("File_paths from dataloader", self.d)
#         input("")

    def __getitem__(self,index):

        self.current_dir = int((index - (index % 155)) / 155)
#         print(index)
#         print(self.current_dir)
# #         print("")
# #         print(self.current_dir)
# #         print(int(self.current_dir))
# #         print("")
# #         input("")
#         print("current directory value", self.current_dir)

        file_t = self.d[self.current_dir] + '/' + self.d[self.current_dir][5:] + "_" + "whimg_norm" + '.nii.gz'
        full_path = self.path + file_t
        img_a = nib.load(full_path)
        img_data = img_a.get_fdata()
        
        if(img_data.ndim == 3):
            img_data = img_data[np.newaxis,:,:,:]
        
        # this equation had a -1 at the end of it which when tested seemed to be doing the wrong thing (i.e. having the final image in a dataset seen twice since the indexing was -1 instead of 0. not sure if this would have actually caused any errors or not but still got it out of the way. (also fixed for the corresponding label index)
        img = img_data[:,:,:,int(index % 155)]
        
        # interpolate image 
        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(img,(int(img.shape[2]*self.size),int(img.shape[3]*self.size)))
        
        #                          image return end                           #
        #######################################################################
        #                         labels return start                         #

        file_label = self.d[self.current_dir] + '/' + self.d[self.current_dir][5:] + "_" + "whseg_norm" + '.nii.gz'
        l_full_path = self.path + file_label
        
        # nibabel code to load the nii.gz file format, change it into a numpy array (Fdata) and then select the correct index
        l_img = nib.load(l_full_path)
        img_labels = l_img.get_fdata()
        label = img_labels[:,:,int(index % 155)]
        
        # interpolate label to change the size of the image based on the given multiplier
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label,(int(label.shape[2]*self.size),int(label.shape[3]*self.size)))
        
        #                          labels return end                          #
        #######################################################################
        
        if self.apply_transform == True:
            img,label = self.Transform(img,label)
            
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        
        return img,label,self.d[self.current_dir]
    
    # a number of transformations to apply to both the labels and the images in tandem (same for each example) for dataset augmentation as a standard feature that can be toggled if necessary. includes:
    # 50% chance to horizontal flip 
    # 50% chance to vertical flip 
    # 25% chance to rotate up to 30 degrees
    # 25% chance to perform a 10% - 20% zoom / scaling around the center
    
    # i dont actually think that this actually works (as in it will apply all examples int 20% of cases and both flips in 50% of cases
    # this is if the random is the same value, though i have put it in as calling for each of the functions so maybe im overthinking this and it is recomputed at each event but might have to print out on an example to find out.
    
    def Transform(self, image, label):

        # 50% horizontal flip 
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 50% vertical flip 
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
        # return the number of value within the loaded data
        # since in this case we are loading on a file-wise basis for the fully formed datasets (with 155 slices per volume) the value from self.d is multipled by that value, but would have to rethink this in the case of a variable dataset such as extracting the final value from the previously calculated index file. need to think about it. but not necessary to change at this time.
        x = len(self.d)# * 155
        return x