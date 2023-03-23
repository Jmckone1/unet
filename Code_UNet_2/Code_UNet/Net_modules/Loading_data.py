from torch.utils.data.dataset import Dataset
import Net_modules.Parameters_SEG as Param
import nibabel as nib
import numpy as np
import random
import torch
import sys
import os

import cv2
import albumentations as A

random.seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
torch.manual_seed(Param.Parameters.PRANO_Net["Global"]["Seed"])

np.set_printoptions(threshold=sys.maxsize)

class Load_Dataset(Dataset):
    def __init__(self,path,image_data, masks_data, transform=None):
        self.path = path
        self.image_folders = image_data
        self.masks_folders = masks_data
        self.transforms = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self,idx):
        
        image_path = os.path.join(self.path,'imagesTr/',self.image_folders[idx] + ".nii.gz")
        img = nib.load(image_path).get_fdata()
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            mask_path = os.path.join(self.path,'BiLabelsTr/', self.masks_folders[idx] + ".npz")
                                     
            numpy_mask = np.load(mask_path)
            label = numpy_mask["RANO"][np.newaxis,:]  
            
            
        else:
            mask_path = os.path.join(self.path,'labelsTr/',self.masks_folders[idx] + ".nii.gz")
            label = nib.load(mask_path).get_fdata() 
            
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Apply_Augmentation"] == True:
            img, label = augmentation(img,label)
            
        return (img,label)
    
    def augmentation(self,img,label):
        
        h_flip = 0.5
        v_flip = 0.5
        crop = 0.5
        rotate = 0.5
        
        # 1/6th of the image as maximum reduction from crop around the center
        min_max = [int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0] - 
                       int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0]/ 6)),
                   int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] - 
                       int(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1] / 6))]

        if Param.Parameters.PRANO_Net["Global"]["Debug"] : print("Augmentation")
        
        # Regression ######################################################################################### #
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            keypoint_format = [
                (label[1],label[0]),
                (label[3],label[2]),
                (label[5],label[4]),
                (label[7],label[6])
            ]
            
            transform1 = A.Compose([
                A.HorizontalFlip(p=h_flip),
                A.VerticalFlip(p=v_flip),
                A.RandomSizedCrop(min_max,
                                  width = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0],
                                  height = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1],
                                  interpolation = cv2.INTER_CUBIC,
                                  p = crop),
                A.RandomRotate90(p=rotate)
                ],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
            transformed = transform1(image=img, keypoints=keypoint_format)
            transformed_img = transformed['image']
            transformed_label = transformed['keypoints']
            
            transformed_label = [transformed_label[0][1], transformed_label[0][0],
                                 transformed_label[1][1], transformed_label[1][0],
                                 transformed_label[2][1], transformed_label[2][0],
                                 transformed_label[3][1], transformed_label[3][0]]
            
        # Segmentation ##################################################################################### #
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
            transform2 = A.Compose([
                A.HorizontalFlip(p=h_flip),
                A.VerticalFlip(p=v_flip),
                A.RandomSizedCrop(min_max,
                                  width = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0],
                                  height = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1],
                                  interpolation = cv2.INTER_CUBIC,
                                  p = crop),
                A.RandomRotate90(p=rotate)
                ])
            
            transformed = transform2(image=img, mask=label)
            transformed_img = transformed['image']
            transformed_label = transformed['mask']

        return transformed_img,transformed_label
