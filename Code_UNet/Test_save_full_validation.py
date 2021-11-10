import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

#from Unet_modules.Brats_dataloader_3 import BraTs_Dataset
from Unet_modules.Full_model_dataloader_main_Copy import BraTs_Dataset

#from Unet_modules.dataloader_test import Test_Dataset
import Net.Unet_components_v2 as net
import csv
from os import walk
import nibabel as nib
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

# this has been written as an external file function and should be called from there instead of here
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
    
    if pred_1.sum() == 0 and truth_1.sum() > 2:
        score = 0
        
    return score

def Test_save(Test_data, unet, unet_opt, path, path_ext, save_path, load_path, save=False, save_image = False, save_val =""):
    
    f = []
    d = []
    DS_mean = []
    DS_mean_all = []
    DS_mean_none = []
    
    if not os.path.exists(save_path) and save == True:
        # Create a new directory because it does not exist 
        os.makedirs(save_path)

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
    DS_all = []
    DS_none = []
    data_val = 0 # number for file output naming
    
    for truth_input in tqdm(Test_data):

            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            
#             label_input = label_input.to(device)
#             label_input = label_input.float()
#             label_input = label_input.squeeze()
            
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()
            
            plt.show()

            pred_output = pred.cpu().detach().numpy()
#             truth_output = label_input.cpu().detach().numpy()

            if save == True:
                for i in range(cur_batch_size):
                    pred_1 = np.clip(pred_output[i,:,:], 0, 1.0)
                    pred_1 = np.where(pred_1 > 0.5, 1, 0)
                    pred_img[:,:,img_num] = pred_1
                    
#                     # DS calculates the dice for all slices with a tumour region in ground truth
#                     # DS_all calculates the dice for all slices including those that are empty in truth and prediction
#                     # DS_none calulates the dice only for slices with no tumour region in the ground truth
#                     if np.where(truth_output[i,:,:] > 0.5, 1, 0).sum() != 0:
#                         DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
#                     else:
#                         DS_none.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                        
#                     DS_all.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                    
                    img_num += 1
                    
                    if img_num == 155:
                        print("image 155 slice")
                        
                        if data_val < HGG_len:
                            ext = path_ext[0]
                        else:
                            ext = path_ext[1]
                                
#                         mean_val = np.mean(DS)
#                         mean_val_all = np.mean(DS_all)
#                         mean_val_none = np.mean(DS_none)
                        
#                         DS_mean.append(mean_val)
#                         DS_mean_all.append(mean_val_all)
#                         DS_mean_none.append(mean_val_none)
#                         DS = []
#                         DS_all = []
#                         DS_none = []
                        
                        if save == True:
                            print("saving: ", save_path + ext + "_" + d[data_val] + '.nii.gz')
                            pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
                            nib.save(pred_img_save, os.path.join(save_path + ext + "_" + d[data_val] + '.nii.gz'))  
#                             pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
#                             nib.save(pred_img_save, os.path.join(save_path + ext + "_" + d[data_val] + '_' + str(int(mean_val*100)) + "_" + save_val +'.nii.gz'))  

                        data_val += 1
                        pred_img = np.empty((240,240,155))
                        img_num = 0
#     if save == True:
#         with open(os.path.join(save_path + "0_Tumour_slice_validation_dice.csv"), 'w') as f:
#             write = csv.writer(f) 
#             write.writerow(DS_mean)
#         with open(os.path.join(save_path + "0_All_slice_validation_dice.csv"), 'w') as f:
#             write = csv.writer(f) 
#             write.writerow(DS_mean_all)
#         with open(os.path.join(save_path + "0_None_slice_validation_dice.csv"), 'w') as f:
#             write = csv.writer(f) 
#             write.writerow(DS_mean_none)


# load_path = ['Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_0.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_frozen/checkpoint_0.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_unfrozen/checkpoint_0.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_1.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_frozen/checkpoint_1.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_unfrozen/checkpoint_1.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_2.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_frozen/checkpoint_2.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_unfrozen/checkpoint_2.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_0_100.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_frozen/checkpoint_0_100.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_unfrozen/checkpoint_0_100.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_0_550.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_frozen/checkpoint_0_550.pth',
#              'Checkpoints/Full_model_MK5_H16_O4A4_unfrozen/checkpoint_0_550.pth']

# save_path = ['Predictions/MK_5_model_predictions_2/MK_5_base_Epoch_0_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_frozen_Epoch_0_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_unfrozen_Epoch_0_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_base_Epoch_1_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_frozen_Epoch_1_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_unfrozen_Epoch_1_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_base_Epoch_2_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_frozen_Epoch_2_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_unfrozen_Epoch_2_validation/',
#              'Predictions/MK_5_model_predictions_2/MK_5_base_step_100/',
#              'Predictions/MK_5_model_predictions_2/MK_5_frozen_step_100/',
#              'Predictions/MK_5_model_predictions_2/MK_5_unfrozen_step_100/',
#              'Predictions/MK_5_model_predictions_2/MK_5_base_step_550/',
#              'Predictions/MK_5_model_predictions_2/MK_5_frozen_step_550/',
#              'Predictions/MK_5_model_predictions_2/MK_5_unfrozen_step_550/']

values = np.linspace(50, 550, num=11)

path = "MICCAI_BraTS_2018_Data_Validation/data"
# path_ext = ["/HGG","/LGG"]
path_ext = [""]

unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

dataset_single = BraTs_Dataset("MICCAI_BraTS_2018_Data_Validation/data", path_ext, size=size, apply_transform=False)
Single_data = DataLoader(
    dataset=dataset_single,
    batch_size=batch_size,
    shuffle=False)

# for i in range(3):
#     for j in range(len(values)):

#         load_path = 'Checkpoints/Full_model_MK5_H16_O4A4_base/checkpoint_' + str(i) + '_' + str(int(values[j])) + '.pth'
#         save_path = 'Predictions/MK_5_model_predictions_2/MK_5_base_step_' + str(i) + '_' + str(int(values[j])) + '/'

#         checkpoint = torch.load(load_path)

#         unet.load_state_dict(checkpoint['state_dict'])
#         unet_opt.load_state_dict(checkpoint['optimizer'])

#         Test_save(Single_data, unet, unet_opt, path, path_ext, save_path, load_path, save=True)

load_path = 'Checkpoints/Full_model_MK5_H16_O0A0_baseline_alldata/checkpoint_2.pth'
save_path = 'Predictions/model_5_offical_val_all_data_baseline_official_main/'

checkpoint = torch.load(load_path)

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

Test_save(Single_data, unet, unet_opt, path, path_ext, save_path, load_path, save=True)
