# from Unet_modules.Full_model_dataloader_main_Copy import BraTs_Dataset
from Unet_modules.Unet_Main_dataloader_test import BraTs_Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import Net.Unet_components_v2 as net
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
from os import walk
import numpy as np
import torch
import csv
import os

from Unet_modules.Evaluation import DiceLoss
import Unet_modules.Parameters_seg as Param

torch.manual_seed(Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=Param.Global.GPU

# BCE with Logits loss, may change to soft dice
# criterion = nn.BCEWithLogitsLoss()
criterion = DiceLoss()

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

# change the import values here to remove the need for HGG and LGG switching- this is important and the code will crash otherwise - with the values not lining up since the dataset rework.

# def Test_save_oval(Test_data, unet, unet_opt, path, path_ext, save_path, load_path, save=False, save_image = False, save_val =""):
    
#     d = []
#     index_max = []
#     index_max.extend([0])
#     HGG_len = 0
#     c_s = 0
#     current_dir = 0
    
#     DS_mean = []
#     DS_mean_all = []
#     DS_mean_none = []
    
#     if not os.path.exists(save_path) and save == True:
#         # Create a new directory because it does not exist 
#         os.makedirs(save_path)
        
#     # each extension - HGG or LGG
#     for input_ in range(len(path_ext)):
#         counter = 0
#         # each folder in extension
#         for files in os.scandir(path + self.path_ext[input_]):
#             if files.is_dir() or files.is_file():
#                 if not files.name.startswith("."):
#                     d.append(files.name)
#         counter = len(d)
#         print(d)

#         if not os.path.exists(path + Param.rData.index_file):
#             print("Creating index_file...")
#             print(path + Param.rData.index_file)
#             for directory in tqdm(range(counter-c_s)):
#                 if directory == 0:
#                     if input_ == 0:
#                         c_s = counter
#                 if input_ == 1:
#                     directory = directory + c_s
                    
#                     file = self.d[directory] + '/' + self.d[directory] + "_" + Param.rData.image_in + '.nii.gz'
#                     full_path = os.path.join(path + path_ext[input_], file)
#                     img_a = nib.load(full_path)
#                     img_data = img_a.get_fdata()

#                     self.index_max.extend([img_data.shape[3] + self.index_max[-1]])
# #                 print(input_)
# #                 print(len(self.path_ext)-1)
#                 if input_ == (len(self.path_ext)-1):
#                     print("Saving index file . . . ")
#                     np.save(path + Param.rData.index_file, self.index_max)
#                     print("Index file complete")
#             else:
#                 self.index_max = np.load(path + Param.rData.index_file)

#                 # value for extension swapping
#                 if input_ == 0:
#                     self.HGG_len = self.index_max[-1]
#     input("")
#     count = self.index_max[-1]
            
#     unet.eval()
    
#     img_num = 0 # file size output
#     pred_img = np.empty((240,240,155)) # prediction array output
#     DS = [] # dice score total array output
#     DS_all = []
#     DS_none = []
# #     data_val = 0 # number for file output naming
    
#     for truth_input in tqdm(Test_data):

#         cur_batch_size = len(truth_input)

#         # flatten ground truth and label masks
#         truth_input = truth_input.to(Param.testNet.device)
#         truth_input = truth_input.float() 
#         truth_input = truth_input.squeeze()

#         # set accumilated gradients to 0 for param update
#         unet_opt.zero_grad()
#         pred = unet(truth_input)
#         pred = pred.squeeze()

#         plt.show()

#         pred_output = pred.cpu().detach().numpy()

#         if save == True:
#             for i in range(cur_batch_size):
#                 pred_1 = np.clip(pred_output[i,:,:], 0, 1.0)
#                 pred_1 = np.where(pred_1 > 0.5, 1, 0)
#                 pred_img[:,:,img_num] = pred_1

#                 img_num += 1

#                 if img_num == 155:
#                     print("image 155 slice")

# #                     if data_val < HGG_len:
# #                         ext = path_ext[0]
# #                     else:
# #                         ext = path_ext[1]

#                     if save == True:
#                         print("saving: ", save_path + ext + "_" + d[data_val] + '.nii.gz')
#                         pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
#                         nib.save(pred_img_save, os.path.join(save_path + ext + "_" + d[data_val] + '.nii.gz'))  

#                     data_val += 1
#                     pred_img = np.empty((240,240,155))
#                     img_num = 0
                    
                    
def Test_save(Test_data, unet, unet_opt, path, path_ext, save_path, load_path, save=False, save_image = False, save_val =""):
    
    d = []
    index_max = []
    index_max.extend([0])
    HGG_len = 0
    c_s = 0
    current_dir = 0
    
    DS_mean = []
    DS_mean_all = []
    DS_mean_none = []
    
    if not os.path.exists(save_path) and save == True:
        # Create a new directory because it does not exist 
        os.makedirs(save_path)
#         os.makedirs(save_path + "/HGG")
#         os.makedirs(save_path + "/LGG")
        
#     # each extension - HGG or LGG
#     for input_ in range(len(path_ext)):
#         counter = 0
#         # each folder in extension
#         for files in os.scandir(path + path_ext[input_]):
#             if files.is_dir() or files.is_file():
#                 if not files.name.startswith("."):
#                     print(path_ext[input_] + "/" + files.name)
#                     d.append(path_ext[input_] + "/" + files.name)
#         counter = len(d)

#         if not os.path.exists(path + Param.sData.index_file )and Param.sData.refresh_index == True:
#             print("Creating index_file...")
#             print(path + Param.sData.index_file)
#             for directory in tqdm(range(counter-c_s)):
#                 if directory == 0:
#                     if input_ == 0:
#                         c_s = counter
#                 if input_ == 1:
#                     directory = directory + c_s
                    
#                     file = d[directory] + '/' + d[directory][5:] + "_" + Param.sData.image_in + '.nii.gz'
#                     full_path = os.path.join(path + path_ext[input_], file)
                    
#                     img_a = nib.load(full_path)
#                     img_data = img_a.get_fdata()

#                     index_max.extend([img_data.shape[3] + index_max[-1]])

#                 if input_ == (len(path_ext)-1):
#                     print("Saving index file . . . ")
#                     np.save(path + Param.sData.index_file, index_max)
#                     print("Index file complete")
#             else:
#                 index_max = np.load(path + Param.sData.index_file)

#                 # value for extension swapping
#                 if input_ == 0:
#                     HGG_len = index_max[-1]
                    
#     print("file name aquisition complete, press Enter to continue . . . ")
# #     input("")
#     count = index_max[-1]
            
    unet.eval()
    
    img_num = 0 # file size output
    pred_img = np.empty((240,240,155)) # prediction array output
    DS = [] # dice score total array output
    DS_all = []
    DS_none = []
    data_val = 0 # number for file output naming
    
    for truth_input, label_input, dataloader_path in tqdm(Test_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.testNet.device)
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()
        
        label_input = label_input.to(Param.testNet.device)
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

                # DS calculates the dice for all slices with a tumour region in ground truth
                # DS_all calculates the dice for all slices including those that are empty in truth and prediction
                # DS_none calulates the dice only for slices with no tumour region in the ground truth
                if np.where(truth_output[i,:,:] > 0.5, 1, 0).sum() != 0:
                    DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                else:
                    DS_none.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))

                DS_all.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))

                img_num += 1
                #print(img_num)

                if img_num == 155:
                    print("image 155 slice")

#                     if data_val < HGG_len:
#                         ext = path_ext[0]
#                     else:
#                         ext = path_ext[1]
                        
                    mean_val = np.mean(DS)
                    mean_val_all = np.mean(DS_all)
                    mean_val_none = np.mean(DS_none)

                    DS_mean.append(mean_val)
                    DS_mean_all.append(mean_val_all)
                    DS_mean_none.append(mean_val_none)
                    DS = []
                    DS_all = []
                    DS_none = []
                    
#                     print("save_path", save_path)
#                     print("#################################")
#                     print("data_path", dataloader_path[i])
#                     print("#################################")
#                     print("addition [4:]", save_path + dataloader_path[i][4:] + '.nii.gz')
#                     input("#################################")
                    
                    if save == True:
                        print("saving: ", save_path + dataloader_path[i][4:] + '.nii.gz')
                        pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
                        nib.save(pred_img_save, os.path.join(save_path + dataloader_path[i][4:] + '.nii.gz')) 
                        

                        pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
                        nib.save(pred_img_save, os.path.join(save_path + dataloader_path[i][4:] + '_' + str(int(mean_val*100)) + "_" + save_val +'.nii.gz'))  
                                                             
                    data_val += 1
                    pred_img = np.empty((240,240,155))
                    img_num = 0
                                                             
    if save == True:
        with open(os.path.join(save_path + "0_Tumour_slice_validation_dice.csv"), 'w') as f:
            write = csv.writer(f) 
            write.writerow(DS_mean)
        with open(os.path.join(save_path + "0_All_slice_validation_dice.csv"), 'w') as f:
            write = csv.writer(f) 
            write.writerow(DS_mean_all)
        with open(os.path.join(save_path + "0_None_slice_validation_dice.csv"), 'w') as f:
            write = csv.writer(f) 
            write.writerow(DS_mean_none)


##################################################################################################################################
# dataset length splitting - currently needs testing - the code above is the prior functioning code ##############################
##################################################################################################################################
print("starting model")

# implement the parameter file funtions for this to work, you dont have a cchoice now mwahahahahahaha

# apply_transform adds data augmentation to the model - in this case we apply horizontal flip, vertical flip, rotation up to 30 degrees and between 10% and 20% zoom to the center of the image; with 50%, 50%, 25% and 25% chances of occuring.
dataset = BraTs_Dataset(Param.SegNet.dataset_path, path_ext = Param.SegNet.extensions, size=Param.testNet.size, apply_transform=False)
print("initialised dataset")

index_f = np.load(Param.SegNet.dataset_path + Param.sData.index_file)
print("loaded index file")
patients_number = len(index_f)

print("length start")
train_length = index_f[int(np.floor(patients_number*Param.SegNet.train_split))]
validation_length = index_f[int(np.ceil(patients_number*Param.SegNet.validation_split))]
test_length = index_f[int(np.ceil(patients_number*Param.SegNet.test_split))-1]
all_data_length = index_f[-1]
custom_split = index_f[int(np.ceil(patients_number*Param.SegNet.custom_split_amount))-1]

print("range start")
train_range = list(range(0,train_length))
val_range = list(range(train_length,train_length+validation_length))
test_range = list(range(train_length+validation_length,train_length+validation_length+test_length))
all_data_range = list(range(0,all_data_length))
custom_split_range = list(range(0,custom_split))

print(train_length)
print(validation_length)
print(all_data_length)

train_data_m = torch.utils.data.RandomSampler(train_range,False)
validation_data_m = torch.utils.data.RandomSampler(val_range,False)
test_data_m = torch.utils.data.SubsetRandomSampler(test_range,False)
all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)



test_data_us = torch.utils.data.SubsetRandomSampler(test_range,False)
print(test_data_us)
print(test_range)
input("")
print("produced dataset split amounts")
##################################################################################################################################

# https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
print("Full_dataset: ", len(all_data_m))
print("Training: ", len(train_data_m))
print("validation: ", len(validation_data_m))

print("")
print("Param values")

for val, var in enumerate(vars(Param.Global)):
    if val == len(vars(Param.Global)) - 3:
        print("")
        break
    else:
        print(var, getattr(Param.Global, var))

for val, var in enumerate(vars(Param.testNet)):
    if val == len(vars(Param.testNet)) - 3:
        print("")
        break
    else:
        print(var, getattr(Param.testNet, var))

input("Press enter to continue . . . . . .")

# Train_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.testNet.batch_size,
#     sampler=train_data_m)

# Val_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.testNet.batch_size,
#     sampler=test_data_m, 
#     shuffle=False,
#     drop_last=False)

Test_data=DataLoader(
    dataset=dataset,
    batch_size=Param.testNet.batch_size,
    sampler= test_range, 
    shuffle=False,
    drop_last=False)

# print("actual validation length", len(Test_data.sampler))

unet = net.UNet(Param.testNet.input_dim, 
                Param.testNet.label_dim, 
                Param.testNet.hidden_dim).to(Param.testNet.device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.testNet.lr, weight_decay=Param.testNet.weight_decay)

values = np.linspace(50, 550, num=11)

if Param.testNet.intermediate_checkpoints == True:
    for j in range(len(values)):

        load_path = Param.testNet.load_path + "/checkpoint_0" + "_" + str(int(values[j])) + ".pth"
        save_path = Param.testNet.save_path + "/step_0_" + str(int(values[j]))

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data, unet, unet_opt, Param.testNet.dataset_path, Param.testNet.extensions, save_path, load_path, save=True)

if Param.testNet.end_epoch_checkpoints == True:
    for i in range(3):

        load_path = Param.testNet.load_path + "/checkpoint_" + str(i) + ".pth"
        save_path = Param.testNet.save_path + "/step_" + str(i)

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data, unet, unet_opt, Param.testNet.dataset_path, Param.testNet.extensions, save_path, load_path, save=True)

print("END")