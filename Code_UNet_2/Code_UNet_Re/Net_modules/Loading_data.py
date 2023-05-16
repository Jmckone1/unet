from torch.utils.data.dataset import Dataset
# import Net_modules.Parameters_SEG as Param
import nibabel as nib
import numpy as np
import random
import torch
import sys
import os

import cv2
import albumentations as A

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
        
#         for i in range(10):
#             i = i + 1
#             x,y = self.__getitem__((i*100) -10)
#             plt.imshow(x[0,:,:])
#             plt.show()
#             plt.imshow(y)
#             plt.show()
            
#             print(np.min(x),np.max(x))
#             print(np.min(y),np.max(y))
            
#             print(x.dtype)
#             print(y.dtype)
            
#             x,y = self.__getitem__((i*100))
#             plt.imshow(x[0,:,:])
#             plt.show()
#             plt.imshow(y)
#             plt.show()
            
#             print(np.min(x),np.max(x))
#             print(np.min(y),np.max(y))
            
#             print(x.dtype)
#             print(y.dtype)
            
#             x,y = self.__getitem__((i*100) +10)
#             plt.imshow(x[0,:,:])
#             plt.show()
#             plt.imshow(y)
#             plt.show()
            
#             print(np.min(x),np.max(x))
#             print(np.min(y),np.max(y))
            
#             print(x.dtype)
#             print(y.dtype)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,idx):
        
        
        
        image_path = os.path.join(self.path,'imagesTr/', self.image_folders[idx] + ".nii.gz")
        
#         print("Image", image_path)
        
        img = nib.load(image_path).get_fdata()
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
#             mask_path = os.path.join(self.path,'BiLabelsTr/', self.masks_folders[idx] + ".npz")
                              
#             # print(mask_path)
#             numpy_mask = np.load(mask_path)
#             label = numpy_mask["RANO"][np.newaxis,:]  
            
#         else:
        mask_path = os.path.join(self.path,'labelsTr/', self.masks_folders[idx] + ".nii.gz")
    
#         print("Mask", mask_path)
        
        label = nib.load(mask_path).get_fdata() 
        
        if self.transform == True:
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Apply_Augmentation"] == True:
            img, label = self.augmentation(img,label)
        
#         print("")
#         print("image and label shape")
#         print(np.shape(img))
#         print(np.shape(label))
        
        return (img,label)
    
    def augmentation(self, img, label):
        
        h_flip = 0.5
        v_flip = 0.5
        crop = 0.5
        rotate = 0.5
        
        # 1/6th of the image as maximum reduction from crop around the center
#         min_max = [int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0] - 
#                        int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0]/ 6)),
#                    int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] - 
#                        int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] / 6))]

        min_max = [int(240 - int(240/ 6)), int(240 - int(240 / 6))]

#         if Param.Parameters.PRANO_Net["Global"]["Debug"] : print("Augmentation")
        
#         # Regression ######################################################################################### #
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
#             keypoint_format = [
#                 (label[0,1],label[0,0]),
#                 (label[0,3],label[0,2]),
#                 (label[0,5],label[0,4]),
#                 (label[0,7],label[0,6])
#             ]
            
#             transform1 = A.Compose([
#                 A.HorizontalFlip(p=h_flip),
#                 A.VerticalFlip(p=v_flip),
#                 A.RandomSizedCrop(min_max,
#                                   width = 240,
#                                   height = 240,
#                                   interpolation = cv2.INTER_CUBIC,
#                                   p = crop),
#                 A.RandomRotate90(p=rotate)
#                 ],
#                 keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
#             if Param.Parameters.PRANO_Net["Hyperparameters"]["Input_dim"] > 1:
#                 img = np.reshape(img,(240,240,1))
            
#                 transformed = transform1(image=img[..., :1], keypoints=keypoint_format)
                
#                 img = np.reshape(img,(1,240,240))
#             else:
#                 if img.ndim == 3:
#                     img = np.reshape(img, (240, 240, 4))
#                 transformed = transform1(image=img, keypoints=keypoint_format)
                
#             transformed_img = transformed['image']
#             if img.ndim == 3:
#                 transformed_img = np.reshape(transformed_img, (4, 240, 240))
#             transformed_label = transformed['keypoints']
            
#             transformed_label = [transformed_label[0][1], transformed_label[0][0],
#                                  transformed_label[1][1], transformed_label[1][0],
#                                  transformed_label[2][1], transformed_label[2][0],
#                                  transformed_label[3][1], transformed_label[3][0]]

#             transformed_label = np.expand_dims(transformed_label, axis=0)
            
        # Segmentation ##################################################################################### #
#         if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:

# so for some reason the Vflip on its own causes the loss to flatline and the Hflip when combined with anything else (rotate or randomcrop) also causes the loss to flatline, removing the Hflip and Vflip seems to have stabilised it for the time being.

#         transform0 = A.Compose([
#             A.HorizontalFlip(p=h_flip)
#             ])
    
#         transform1 = A.Compose([
#             A.VerticalFlip(p=v_flip)
#             ])
        
        
#         transform2 = A.Compose([
#             A.RandomSizedCrop(min_max,
#                               width = 240,
#                               height = 240,
#                               interpolation = cv2.INTER_CUBIC,
#                               p = crop)
#             ])
        
#         transform3 = A.Compose([
#             A.RandomRotate90(p=rotate)
#             ])
        
#         transformhr = A.Compose([
#             A.HorizontalFlip(p=h_flip),
#             A.RandomRotate90(p=rotate)
#             ])
        
        transformcr = A.Compose([
            A.RandomSizedCrop(min_max,
                              width = 240,
                              height = 240,
                              interpolation = cv2.INTER_CUBIC,
                              p = crop),
            A.RandomRotate90(p=rotate)
            ])

#         if 4 > 1:
        img = np.reshape(img,(240,240,4))

        transformed = transformcr(image=img, mask=label)

        transformed_img = transformed['image']
        transformed_img = np.reshape(transformed_img,(4,240,240))

#         else:
#             transformed = transform2(image=img, mask=label)

#             transformed_img = transformed['image']
        transformed_label = transformed['mask']

        return transformed_img,transformed_label
