import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np
from Unet_modules.dataloader_test import Test_Dataset
import Net.Unet_components as net
from os import walk
import nibabel as nib
import os

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
criterion = nn.BCEWithLogitsLoss()

n_epochs = 3
input_dim = 4
label_dim = 1
hidden_dim = 16

display_step = 50
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(240 * size)
device = 'cuda'

def Test(Test_data, unet, unet_opt, path, path_ext=["/data"]):
    
    f = []
    d = []
    
    counter = 0
    # each folder in extension
    for (dir_path, dir_names, file_names) in walk(path + path_ext[0]):

        # gets rid of any pesky leftover .ipynb_checkpoints files
        if not dir_names == []:
            if not dir_names[0].startswith("."):

                f.extend(file_names)
                d.extend(dir_names)
                counter = len(d)

        HGG_len = counter * 155
        #print(HGG_len)

    #print(path + path_ext[0])

    unet.eval()
    
    img_num = 0 # file size output
    pred_img = np.empty((240,240,155)) # prediction array output
    data_val = 0 # number for file output naming
    
    for truth_input in tqdm(Test_data):

            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()

            pred_output = pred.cpu().detach().numpy()
            
            for i in range(cur_batch_size):
                pred_1 = np.clip(pred_output[i,:,:], 0, 1.0)
                pred_1 = np.where(pred_1 > 0.5, 1, 0)
                pred_img[:,:,img_num] = pred_1

                img_num += 1

                if img_num == 155:

                    ext = path_ext[0]

                    pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
                    nib.save(pred_img_save, os.path.join('Predictions' + ext + "_" + d[data_val] + '.nii.gz'))  

                    data_val += 1
                    pred_img = np.empty((240,240,155))
                    img_num = 0
                    
dataset = Test_Dataset("MICCAI_BraTS_2018_Data_Validation",path_ext=["/data"],size=size,apply_transform=False)

Data_1 = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False)


unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

checkpoint = torch.load("Checkpoints/Checkpoints model_3/checkpoint_2.pth")

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

Test(Data_1, unet, unet_opt,"MICCAI_BraTS_2018_Data_Validation")