import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

#from Unet_modules.Brats_dataloader_3 import BraTs_Dataset
from Unet_modules.Full_model_dataloader_main import BraTs_Dataset

#from Unet_modules.dataloader_test import Test_Dataset
import Net.Unet_components_v2 as net
import csv
from os import walk
import nibabel as nib
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

load_path = "Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_0_550.pth"
save_path = 'Predictions/MK_5_model_predictions/MK_5_base_step_550/'

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
criterion = nn.BCEWithLogitsLoss()

n_epochs = 4
input_dim = 4
label_dim = 1
hidden_dim = 16

display_step = 100
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(240 * size)
device = 'cuda'

def dice_score(prediction, truth):
    # clip changes negative vals to 0 and those above 1 to 1
    pred_1 = np.clip(prediction, 0, 1.0)
    truth_1 = np.clip(truth, 0, 1.0)

    # binarize
    pred_1 = np.where(pred_1 > 0.5, 1, 0)
    truth_1 = np.where(truth_1 > 0.5, 1, 0)

    # Dice calculation
    product = np.dot(truth_1.flatten(), pred_1.flatten())
    dice_num = 2 * product + 1
    pred_sum = pred_1.sum()
    label_sum = truth_1.sum()
    dice_den = pred_sum + label_sum + 1
    score = dice_num / dice_den

    return score

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28),title=""):

    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.title(title)
    plt.imshow((image_grid.permute(1, 2, 0).squeeze()* 255).type(torch.uint8))
    plt.show()

def Test_save(Test_data, unet, unet_opt, path, path_ext, save=False, save_val =""):
    f = []
    d = []
    DS_mean = []

    #path = "Brats_2018 data"

    # each extension - HGG or LGG
    for input_ in range(len(path_ext)):
        counter = 0
        # each folder in extension
        for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
                            
            # gets rid of any pesky leftover .ipynb_checkpoints files
            if not dir_names == []:
                if not dir_names[0].startswith("."):
                    
                    f.extend(file_names)
                    d.extend(dir_names)
                    counter = len(d)
                    print(d)
                    print(f)

        # value for extension swapping
        if input_ == 0:
            HGG_len = counter * 155
            
    unet.eval()
    
    img_num = 0 # file size output
    pred_img = np.empty((240,240,155)) # prediction array output
    DS = [] # dice score total array output
    data_val = 0 # number for file output naming
    
    for truth_input, label_input in tqdm(Test_data):

            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            
            label_input = label_input.to(device)
            label_input = label_input.float()
            label_input = label_input.squeeze()
            
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()
            
            plt.show()

            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()

            if save == True:
                for i in range(cur_batch_size):
                    pred_1 = np.clip(pred_output[i,:,:], 0, 1.0)
                    pred_1 = np.where(pred_1 > 0.5, 1, 0)
                    pred_img[:,:,img_num] = pred_1
                    #pred_img[:,:,img_num] = pred_output[i,:,:]
                    
                    DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                    
                    img_num += 1
                    
                    if img_num == 155:
                        
                        if data_val < HGG_len:
                            ext = path_ext[0]
                        else:
                            ext = path_ext[1]
                                
                        mean_val = np.mean(DS)
                        DS_mean.append(mean_val)
                        DS = []
                        
                        pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
                        nib.save(pred_img_save, os.path.join(save_path + ext + "_" + d[data_val] + '_' + str(int(mean_val*100)) + "_" + save_val +'.nii.gz'))  
                        
                        data_val += 1
                        pred_img = np.empty((240,240,155))
                        img_num = 0
                        
    with open(os.path.join(save_path + "validation_dice.csv"), 'w') as f:
        write = csv.writer(f) 
        write.writerow(DS_mean)

path = "Brats_2018_data_split/Validation"
path_ext = ["/HGG","/LGG"]

#unet = net.UNet.load_weights(input_dim, label_dim, hidden_dim,"Checkpoints_RANO/Unet_H16_M8/checkpoint_49.pth").to(device)
unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

checkpoint = torch.load(load_path)

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

dataset_single = BraTs_Dataset("Brats_2018_data_split/Validation", path_ext, size=size, apply_transform=False)
Single_data = DataLoader(
    dataset=dataset_single,
    batch_size=batch_size,
    shuffle=False)

Test_save(Single_data, unet, unet_opt, path, path_ext, save=True)