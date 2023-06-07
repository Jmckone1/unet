from torch.utils.data.dataset import Dataset
import Net_modules.Model_hyperparameters as Param
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import random
import torch
import sys
import os

import cv2
import torchvision.transforms.functional as TF
import torchvision

print(Param.Parameters)

# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)
# torch.cuda.manual_seed(0)

# torch.backends.cudnn.deterministic=True

seed = 11

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

np.set_printoptions(threshold=sys.maxsize)

class Load_Dataset(Dataset):
    def __init__(self,path,image_data, masks_data, transform=True):
        self.path = path
        self.image_folders = image_data
        self.masks_folders = masks_data
        self.transform = transform
        self.val = 0

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,idx):
        
        image_path = os.path.join(self.path,'imagesTr/', 
                                  self.image_folders[idx] + ".nii.gz")
        img = nib.load(image_path).get_fdata()
        mask_path = os.path.join(self.path,'labelsTr/', 
                                 self.masks_folders[idx] + ".nii.gz")
        mask = nib.load(mask_path).get_fdata()         
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)   
        
        if Param.Parameters.Network["Global"]["Debug"] == True:
            fig, ax = plt.subplots(2,2)

            ax[0,0].imshow(img[0,0,:,:], cmap='gray')
            ax[0,0].imshow(mask[0,0,:,:], cmap='jet', alpha=0.5)

            ax[1,0].imshow(img[0,1,:,:], cmap='gray')
            ax[1,0].imshow(mask[0,0,:,:], cmap='jet', alpha=0.5)

            ax[0,1].imshow(img[0,2,:,:], cmap='gray')
            ax[0,1].imshow(mask[0,0,:,:], cmap='jet', alpha=0.5)

            ax[1,1].imshow(img[0,3,:,:], cmap='gray')
            ax[1,1].imshow(mask[0,0,:,:], cmap='jet', alpha=0.5)

            plt.show()
            
            print(np.shape(img))
            print(np.shape(mask))
            print("INDEX", idx)
            
#             if self.val == 50:
#                 input("")
#                 self.val = 0

#             self.val += 1
        
        
#         if self.transform == True:
#             img, mask = self.Transform(img,mask)
            
#         print(idx)

        return (img,mask)
    
#     def Transform(self, image, mask):

#         # 25% horizontal flip 
#         if random.random() > 0.5:
#             image = TF.hflip(image)
#             mask = TF.hflip(mask)

#         # 25% vertical flip 
#         if random.random() > 0.5:
#             image = TF.vflip(image)
#             mask = TF.vflip(mask)

#         # rotation up to 30 degrees
#         if random.random() > 0.25:
#             rotation = random.randint(1,30)
#             image = TF.rotate(image,rotation)
#             mask = TF.rotate(mask,rotation)

#         # 10% - 20% zoom / scaling around the center
#         if random.random() > 0.25:
#             size = Param.Parameters.Network["Hyperparameters"]["Image_size"]
#             resize = random.randint(int(size[0]*0.1),int(size[1]*0.2))
#             crop = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size[0]-resize)])
            
#             image = crop(image)
#             image = TF.resize(image,size[0])

#             mask = crop(mask)
#             msak = TF.resize(mask,size[0])

#         return image, mask