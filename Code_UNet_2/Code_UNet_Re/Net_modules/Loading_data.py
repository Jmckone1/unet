from torch.utils.data.dataset import Dataset
# import Net_modules.Parameters_SEG as Param
import nibabel as nib
import numpy as np
import random
import torch
import sys
import os

import cv2
# import albumentations as A
import torchvision.transforms.functional as TF
import torchvision

random.seed(0)
torch.manual_seed(0)

np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

class Load_Dataset(Dataset):
    def __init__(self,path,image_data, masks_data, transform=True):
        self.path = path
        self.image_folders = image_data
        self.masks_folders = masks_data
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,idx):
        
#         print(idx)

        image_path = os.path.join(self.path,'imagesTr/', self.image_folders[idx] + ".nii.gz")
#         print(image_path)
        img = nib.load(image_path).get_fdata()
        
        
        mask_path = os.path.join(self.path,'labelsTr/', self.masks_folders[idx] + ".nii.gz")

        label = nib.load(mask_path).get_fdata() 
        
#         if self.transform == True:
#             img, label = self.augmentation(img,label)
            
        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)    
        if self.transform == True:
            img, label = self.Transform(img,label)
        return (img,label)
    
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
            
#         image = np.reshape(image,(4,240,240))
        
        return image, label
    
    
    
    
#     def augmentation(self, img, label):
        
#         h_flip = 0.5
#         v_flip = 0.5
#         crop = 0.5
#         rotate = 0.5
        
#         # 1/6th of the image as maximum reduction from crop around the center
# #         min_max = [int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0] - 
# #                        int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0]/ 6)),
# #                    int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] - 
# #                        int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] / 6))]

#         min_max = [int(240 - int(240/ 6)), int(240 - int(240 / 6))]

# #         if Param.Parameters.PRANO_Net["Global"]["Debug"] : print("Augmentation")
        
# #         # Regression ######################################################################################### #
# #         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
# #             keypoint_format = [
# #                 (label[0,1],label[0,0]),
# #                 (label[0,3],label[0,2]),
# #                 (label[0,5],label[0,4]),
# #                 (label[0,7],label[0,6])
# #             ]
            
# #             transform1 = A.Compose([
# #                 A.HorizontalFlip(p=h_flip),
# #                 A.VerticalFlip(p=v_flip),
# #                 A.RandomSizedCrop(min_max,
# #                                   width = 240,
# #                                   height = 240,
# #                                   interpolation = cv2.INTER_CUBIC,
# #                                   p = crop),
# #                 A.RandomRotate90(p=rotate)
# #                 ],
# #                 keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
# #             if Param.Parameters.PRANO_Net["Hyperparameters"]["Input_dim"] > 1:
# #                 img = np.reshape(img,(240,240,1))
            
# #                 transformed = transform1(image=img[..., :1], keypoints=keypoint_format)
                
# #                 img = np.reshape(img,(1,240,240))
# #             else:
# #                 if img.ndim == 3:
# #                     img = np.reshape(img, (240, 240, 4))
# #                 transformed = transform1(image=img, keypoints=keypoint_format)
                
# #             transformed_img = transformed['image']
# #             if img.ndim == 3:
# #                 transformed_img = np.reshape(transformed_img, (4, 240, 240))
# #             transformed_label = transformed['keypoints']
            
# #             transformed_label = [transformed_label[0][1], transformed_label[0][0],
# #                                  transformed_label[1][1], transformed_label[1][0],
# #                                  transformed_label[2][1], transformed_label[2][0],
# #                                  transformed_label[3][1], transformed_label[3][0]]

# #             transformed_label = np.expand_dims(transformed_label, axis=0)
            
#         # Segmentation ##################################################################################### #
# #         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:

# # so for some reason the Vflip on its own causes the loss to flatline and the Hflip when combined with anything else (rotate or randomcrop) also causes the loss to flatline, removing the Hflip and Vflip seems to have stabilised it for the time being.

# #         transform0 = A.Compose([
# #             A.HorizontalFlip(p=h_flip)
# #             ])
    
# #         transform1 = A.Compose([
# #             A.VerticalFlip(p=v_flip)
# #             ])
        
        
# #         transform2 = A.Compose([
# #             A.RandomSizedCrop(min_max,
# #                               width = 240,
# #                               height = 240,
# #                               interpolation = cv2.INTER_CUBIC,
# #                               p = crop)
# #             ])
        
# #         transform3 = A.Compose([
# #             A.RandomRotate90(p=rotate)
# #             ])
        
# #         transformhr = A.Compose([
# #             A.HorizontalFlip(p=h_flip),
# #             A.RandomRotate90(p=rotate)
# #             ])
        
#         transformcr = A.Compose([
#             A.RandomSizedCrop(min_max,
#                               width = 240,
#                               height = 240,
#                               interpolation = cv2.INTER_CUBIC,
#                               p = crop),
#             A.RandomRotate90(p=rotate)
#             ])

# #         if 4 > 1:
#         img = np.reshape(img,(240,240,4))

#         transformed = transformcr(image=img, mask=label)

#         transformed_img = transformed['image']
#         transformed_img = np.reshape(transformed_img,(4,240,240))

# #         else:
# #             transformed = transform2(image=img, mask=label)

# #             transformed_img = transformed['image']
#         transformed_label = transformed['mask']

#         return transformed_img,transformed_label
