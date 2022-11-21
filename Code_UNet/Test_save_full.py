###############################################
# This files import requirements : 
# Unet_modules/Unet_Main_dataloader_test_02
# Unet_modules/Evaluation
# Unet_modules/Parameters_seg
# Net/Unet_components_v2
###############################################

# I have now fixed the files allocation to be randomised.
# Now I will to apply this same methodogoloy to the training and try again so that each is correctly randomised.
# Double check with james but i may just have to leave this here without doing that.
# The main justification for this that the current prediction data only contains LGG volumes.
# Whereas the training and validation contain a mix of both HGG (primarily) and LGG (those that arent in the testing).

# I will also test the model fully on the existing LGG heavy data at the moment now that 
# This works properly so we can make that decision later

from Unet_modules.Unet_Main_dataloader_test_02 import BraTs_Dataset
from torch.utils.data import DataLoader
import Net.Unet_components_v2 as net
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
import numpy as np
import torch
import csv
import os

# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from os import walk

from Unet_modules.Evaluation import DiceLoss
import Unet_modules.Parameters_seg as Param

torch.manual_seed(Param.Global.Seed)
np.random.seed(seed=Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=Param.Global.GPU

# BCE with Logits loss, and Dice loss defintions
# criterion = nn.BCEWithLogitsLoss()
criterion = DiceLoss()

# defining a dictionary class to save the output files.
class Define_dictionary(dict):
    def __init__(self):
        self = dict()
    def add(self, key, value):
        self[key] = value
        
# Function to define the dice score of a volume - taken and adapted from the deepmind project code by recommendation of project supervisor.
def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice

# Test save main function for testing the volumes for output and then saving all the values based on the param file.
def Test_save(Test_data, unet, unet_opt, path, path_ext, save_path, load_path, save=False, save_image = False, save_val =""):
    
    Dice_output = Define_dictionary()
    volume_predi = np.empty((240,240,155))
    volume_predi_nonbinary = np.empty((240,240,155))
    volume_truth = np.empty((240,240,155))
    
    # number for file output naming (data_val) and file size output (img_num)
    data_val = 0 
    img_num = 0 
    
    # create the save path if it doesnt exist
    if not os.path.exists(save_path) and save == True:
        os.makedirs(save_path)
        
    if not os.path.exists(save_path + "/nb") and save == True:
        os.makedirs(save_path + "/nb")
        
    # start test 
    unet.eval()
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
        
        # prodcue predictions
        pred = unet(truth_input)
        pred = pred.squeeze()

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        if save == True:
            # for each batch tested
            for i in range(cur_batch_size):
                if img_num == 0:
                    print(dataloader_path[i])
                
                # appy sigmoid to the predictions which isnt already done
                # check if this is an issue when training.
                prediction_sigmoid = np.clip(pred_output[i,:,:], 0, 1.0)
                prediction_sigmoid = np.where(prediction_sigmoid > 0.5, 1, 0)
                
                # convert from batch format to volume format
                volume_predi[:,:,img_num] = prediction_sigmoid[:,:]
                volume_predi_nonbinary[:,:,img_num] = pred_output[i,:,:]
                volume_truth[:,:,img_num] = truth_output[i,:,:]
                
                img_num += 1
                
                # for each full volume
                if img_num == 155:

                    # calculate dice and define file path for saving
                    print("Full: ", dataloader_path[i])
                    print("Concat: ", dataloader_path[i][5:])
                    print(dataloader_path[i][1:4])
                    volume_dice_output = calculate_dice(volume_predi,volume_truth)
                    print(volume_dice_output)

                    # Add the volume path and the volume dice score to the dictionary
                    Dice_output.add(dataloader_path[i], volume_dice_output)
                    
                    # save image prediction
                    if save == True:
                        print("saving: ", save_path + dataloader_path[i][4:] + '.nii.gz')
                        pred_img_save = nib.Nifti1Image(volume_predi, np.eye(4))
                        nib.save(pred_img_save, os.path.join(save_path + dataloader_path[i][4:] + '.nii.gz')) 
                        
                        pred_img_save_nonbinary = nib.Nifti1Image(volume_predi_nonbinary, np.eye(4))
                        nib.save(pred_img_save_nonbinary, os.path.join(save_path + "/nb" + dataloader_path[i][4:] + '.nii.gz')) 
                                         
                    volume_predi = np.empty((240,240,155))
                    volume_predi_nonbinary = np.empty((240,240,155))
                    volume_truth = np.empty((240,240,155))
                    data_val += 1 
                    img_num = 0
            
            # saving the volume name alongside the volume output dice scores for the predictions
            with open(os.path.join(save_path + "_Dice_predictions.csv"), 'w', encoding='UTF8') as f:
                for key, val in Dice_output.items():
                    writer = csv.writer(f)
                    writer.writerow([key, val])  
    
##################################################################################################################################

print("starting model")

# Apply_transform adds data augmentation to the model
# In this case we apply:
#       50% chance - horizontal flip, 
#       50% chance - vertical flip, 
#       25% chance - rotation up to 30 degrees and between 10%,
#       25% chance - 20% zoom to the center of the image; 

dataset = BraTs_Dataset(Param.testNet.dataset_path, path_ext = Param.testNet.extensions, size=Param.testNet.size, apply_transform=False)
print("initialised dataset")

patients_number = len(dataset)
print("Patients", patients_number)
print("Data", len(dataset))

index_f = np.dot(list(range(0,len(dataset)-1)),155)
dataset_shuffle = np.dot(list(range(0,len(dataset)-1)),155)
np.random.shuffle(dataset_shuffle)
dataset_output = []

for i in range(len(dataset_shuffle)):
    dataset_output = dataset_output + list(range(dataset_shuffle[i],dataset_shuffle[i]+155))

##################
# original values #
###################

print("Length Start")

index_f = np.dot(list(range(0,len(dataset)-1)),155)

train_length = index_f[int(np.floor(patients_number*Param.testNet.train_split))]
validation_length = index_f[int(np.ceil(patients_number*Param.testNet.validation_split))]
test_length = index_f[int(np.ceil(patients_number*Param.testNet.test_split))-1]
all_data_length = index_f[-1]
custom_split = index_f[int(np.floor(patients_number*Param.testNet.custom_split_amount))-2]

print("Range Start")
train_range = list(range(0,train_length))
val_range = list(range(train_length,train_length+validation_length))
test_range = list(range(train_length+validation_length,train_length+validation_length+test_length))
all_data_range = list(range(0,all_data_length))
custom_split_range = list(range(0,custom_split))

print("input stop here")
###################
# better shuffles #
###################

# index_f = np.dot(list(range(0,len(dataset)-1)),155)
# dataset_shuffle = np.dot(list(range(0,len(dataset)-1)),155)
# np.random.shuffle(dataset_shuffle)
# dataset_output = []

# for i in range(len(dataset_shuffle)):
#     dataset_output = dataset_output + list(range(dataset_shuffle[i],dataset_shuffle[i]+155))

# print("Length Start")
# train_length = index_f[int(np.floor(patients_number*Param.testNet.train_split))]
# validation_length = index_f[int(np.ceil(patients_number*Param.testNet.validation_split))]
# test_length = index_f[int(np.ceil(patients_number*Param.testNet.test_split))-1]
# all_data_length = index_f[-1]
# custom_split = index_f[int(np.floor(patients_number*Param.testNet.custom_split_amount))-2]

# train_range_shuffle = dataset_output[0:train_length]
# val_range_shuffle = dataset_output[train_length:train_length+validation_length]
# test_range_shuffle = dataset_output[train_length+validation_length:train_length+validation_length+test_length]
# all_data_range_shuffle = dataset_output[0:all_data_length]
# custom_split_range_shuffle = dataset_output[0:custom_split]

# print(train_length)
# print(validation_length)
# print(all_data_length)

# train_data_m = torch.utils.data.RandomSampler(train_range_shuffle,False)
# validation_data_m = torch.utils.data.RandomSampler(val_range_shuffle,False)
# all_data_m = torch.utils.data.RandomSampler(all_data_range_shuffle,False)
# custom_split_m = torch.utils.data.RandomSampler(custom_split_range_shuffle,False)

# test_data_m = test_range_shuffle
# print("produced dataset split amounts")

##################################################################################################################################

# print("Full_dataset: ", len(all_data_m))
# print("Training: ", len(train_data_m))
# print("validation: ", len(validation_data_m))

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

unet = net.UNet(Param.testNet.input_dim, 
                Param.testNet.label_dim, 
                Param.testNet.hidden_dim).to(Param.testNet.device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.testNet.lr, weight_decay=Param.testNet.weight_decay)


Save = True

# save for intermediate checkpoints defined between 50 and 550 at each 50 batches
if Param.testNet.intermediate_checkpoints == True:
    values = np.linspace(50, 550, num=11)
    for j in range(len(values)):
        
        print(" ")
        print("Starting prediction of qualitative outputs for batch ", values[j])
        
        load_path = Param.testNet.load_path + "/checkpoint_0" + "_" + str(int(values[j])) + ".pth"
        save_path = Param.testNet.save_path + "/step_0_" + str(int(values[j]))

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data, unet, unet_opt, Param.testNet.dataset_path, Param.testNet.extensions, save_path, load_path, save=Save)

# save for at the checkpoints at each end of epoch (in this case 1, 2 and 3)
if Param.testNet.end_epoch_checkpoints == True:
    for i in range(3):
        
        print(" ")
        print("Starting prediction of qualitative outputs for end of Epoch ", i + 1)

        load_path = Param.testNet.load_path + "/checkpoint_" + str(i) + ".pth"
        save_path = Param.testNet.save_path + "/step_" + str(i)

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data, unet, unet_opt, Param.testNet.dataset_path, Param.testNet.extensions, save_path, load_path, save=Save)

print("END")