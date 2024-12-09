# from Net_modules.Unet_Main_dataloader import BraTs_Dataset
from Net_modules.Loading_data import Load_Dataset
import Net_modules.Model_hyperparameters as Param

from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import DiceLoss
import Net.Unet_components_split as net

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
import numpy as np
import shutil
import torch
import csv
import os

import random

print("Loading seed . . .")
np.random.seed(0)#Param.Parameters.Network["Global"]["Seed"])
torch.manual_seed(0)#Param.Parameters.Network["Global"]["Seed"])
random.seed(0)#Param.Parameters.Network["Global"]["Seed"])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(Param.Parameters.Network["Global"]["GPU"])

torch.cuda.empty_cache()

os.chdir(os.getcwd())

headers = ["Global","Hyperparameters","Train_paths"]
print("Torch version: ",torch.__version__)
print("")
for h in headers:
    for key, value in Param.Parameters.Network[h].items():
        print(f'{key: <30}{str(value): <35}')
        
np.set_printoptions(precision=4)

input("Press Enter to continue . . . ")

criterion = nn.BCEWithLogitsLoss()
# criterion = DiceLoss()
#update file read in counter based on the UNet_RANO_cosine code
#update dataloader based on the code_UNet_RANO dataloaders

#--------------------------------------------------------#
#              Define validation start                   #

# for some reason this started taking up to 10x as long per batch to run validation over training - not really ure why, cleaned it up as much as i can but. . . . . . . . . . . .
def Validate(unet, criterion, Val_data, epoch, step = ""):

    print(" ")
    print("Validation...")
    
    unet.eval()
    
    running_loss = 0.0
    cur_step = 0
    
    for truth_input, label_input in tqdm(Val_data):
        DS = []
        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
        # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before validation and calculating final loss values.
        
        if(truth_input.ndim == 3):
            truth_input = truth_input[np.newaxis,:,:,:]
            label_input = label_input[np.newaxis,:,:]

#         if(truth_input.ndim == 3):
#             truth_input = truth_input[:,np.newaxis,:,:]
#             label_input = label_input[:,np.newaxis,:]
            
        pred = unet(truth_input)

        pred = pred.squeeze()
        
        if(pred.ndim == 2):
            pred = pred[np.newaxis,:,:]

        loss = criterion(pred, label_input)
        running_loss =+ loss.item()
        
#         print(running_loss)
        cur_step += 1
        
        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        
        if np.shape(pred_output)[0] == 16:
            fig, ax = plt.subplots(4,4)
            print_counter = 0
            for i in range(4):
                for j in range(4):

                    ax[i,j].imshow(pred_output[print_counter,:,:],cmap='gray')
                    print_counter += 1
            plt.show()
        
        # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before evaluation.
        if(pred_output.ndim == 2):
            pred_output = pred_output[np.newaxis,:,:]
            truth_output = truth_output[np.newaxis,:,:]
        
        for Batch in range(cur_batch_size):
            DS.append(Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:]))
        
        with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + step + "validation_loss.csv", 'a') as f: 
            np.savetxt(f, [running_loss], delimiter=',')
        with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + step + "validation_dice.csv", 'a') as f: 
            np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                
    print("Validation complete")
    print(" ")

    # return losses, DS

#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    
    sigmoid_act = nn.Sigmoid()
    
    # run UNet.load_weights for loading of frozen or unfrozen models, use UNet for no initialisation.
    # if using UNet.load_weights allow_update = False for Frozen weights, allow_update = True for unfrozen weights
    
    if Param.Parameters.Network["Hyperparameters"]["Use_weights"] == True:
        unet = net.UNet.load_weights(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Hidden_dim"], 
                                     Param.Parameters.Network["Train_paths"]["Checkpoint_load"], 
                                     Param.Parameters.Network["Hyperparameters"]["Allow_update"]).to(Param.Parameters.Network["Global"]["device"])
    else:
        unet = net.UNet(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Hidden_dim"]).to(Param.Parameters.Network["Global"]["device"])
        
    unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.Parameters.Network["Hyperparameters"]["Learning_rate"], weight_decay=Param.Parameters.Network["Hyperparameters"]["Weight_decay"])
    
    if not os.path.exists("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture"):

        #####################################
        os.makedirs("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture")
    
    with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture.txt", 'w') as write: 
        write.write("left_path: " + Param.Parameters.Network["Train_paths"]["Checkpoint_load"] + "\n")
        write.write("epochs: " + str(Param.Parameters.Network["Hyperparameters"]["Epochs"]) + "\n")
        write.write("batch size: " + str(Param.Parameters.Network["Hyperparameters"]["Batch_size"]) + "\n")
        write.write("learning rate: " + str(Param.Parameters.Network["Hyperparameters"]["Learning_rate"]) + "\n")
        write.write(str(unet))
        
    original = Param.Parameters.Network["Global"]["Param_location"]
    target = "Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_hyperparameters.py"
    if not os.path.exists("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
        os.makedirs("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"])
        
    shutil.copyfile(original, target)

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #

    for epoch in range(Param.Parameters.Network["Hyperparameters"]["Epochs"]):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss = 0.0

        for truth_input, label_input in tqdm(Train_data, desc= running_loss):
            
            DS = []
            
            cur_batch_size = len(truth_input)

            truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
            label_input = label_input.float()
            label_input = label_input.squeeze()

            # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before validation and calculating final loss values.
            
            if(truth_input.ndim == 3):
                truth_input = truth_input[:,np.newaxis,:,:]
                label_input = label_input[:,np.newaxis,:]
                
                
#             print(truth_input.ndim)
#             print(np.shape(truth_input))
#             print("")
#             print(label_input.ndim)
#             print(np.shape(label_input))
#             input("enter")
        
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
#             pred = pred.squeeze()
            
            if(pred.ndim == 2):
                pred = pred[np.newaxis,:,:]
            
            # forward
            
#             print(np.shape(pred),np.shape(label_input))
#             input("This here is the place")
            label_input = label_input[:,np.newaxis,:,:]
            unet_loss = criterion(pred, label_input)
            
            # backward
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item()
#             print("Loss", running_loss)
            # loss_values.append(running_loss)
            cur_step += 1
            
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            
#             pred_output = torch.sigmoid(pred_output)
            
#             if np.shape(pred_output)[0] == 16:
#                 fig, ax = plt.subplots(4,4)
#                 print_counter = 0
#                 for i in range(4):
#                     for j in range(4):

#                         ax[i,j].imshow(pred_output[print_counter,:,:],cmap='gray')
#     #                     print(np.min(pred_output[print_counter,:,:]),np.max(pred_output[print_counter,:,:]))
#                         print_counter += 1
#                 plt.show()
            
            
            # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before evaluation.
            if(pred_output.ndim == 2):
                pred_output = pred_output[np.newaxis,:,:]
                truth_output = truth_output[np.newaxis,:,:]
            dice_val = 0
            for Batch in range(cur_batch_size):
                dice_val = Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:])
                
                DS.append(dice_val)
#             print(dice_val)
            with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + "training_loss.csv", 'a') as f: 
                np.savetxt(f, [running_loss], delimiter=',')
            with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + "training_dice.csv", 'a') as f: 
                np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                
#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            # for each display step in the epoch (each 50 by default)
            if cur_step % Param.Parameters.Network["Hyperparameters"]["Batch_display_step"] == 0:
                # only for the first epoch and only before step 550
                # this is primarily for the memory concerns of the project, 
                # 550 itself is a arbitrary value chosen becuase of plotting sizes.
                if epoch == 0 and cur_step <= 550:
                    checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                    out = "Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                    torch.save(checkpoint, out)

                    # if enabled this will validate the model duing every *DISPLAY STEP* (default 50 batches) during the first epoch

                    if Param.Parameters.Network["Hyperparameters"]["Evaluate"] == True:
                        if epoch == 0:
                            Validate(unet, criterion, Val_data, epoch, step = "_" + str(cur_step))

        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)

        Validate(unet, criterion, Val_data, epoch)

    print('Finished Training Dataset')

# #####################################################################################################
# # dataset length splitting - currently needs testing - the code above is the prior functioning code #
# #####################################################################################################
# print("starting model")

# # apply_transform adds data augmentation to the model - in this case we apply horizontal flip, vertical flip, rotation up to 30 degrees and between 10% and 20% zoom to the center of the image; with 50%, 50%, 25% and 25% chances of occuring.
# dataset = BraTs_Dataset(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"],
#                         path_ext = Param.Parameters.Network["Train_paths"]["Extensions"],
#                         size= Param.Parameters.Network["Hyperparameters"]["Image_scale"], 
#                         apply_transform=Param.Parameters.Network["Hyperparameters"]["Apply_Augmentation"])
# print("initialised dataset")

# index_f = np.load(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"] + Param.Parameters.Network["Old_Hyperparameters"]["Index_File"])
# print("loaded index file")
# patients_number = len(index_f)

# print("length start")
# train_length = index_f[int(np.floor(patients_number*Param.Parameters.Network["Hyperparameters"]["Train_split"]))-1]
# validation_length = index_f[int(np.ceil(patients_number*Param.Parameters.Network["Hyperparameters"]["Validation_split"]))-1]
# test_length = index_f[int(np.ceil(patients_number*Param.Parameters.Network["Hyperparameters"]["Test_split"]))-1]
# all_data_length = index_f[-1]
# custom_split = index_f[int(np.ceil(patients_number*Param.Parameters.Network["Hyperparameters"]["Custom_split"]))-1]

# print("range start")
# train_range = list(range(0,train_length))
# val_range = list(range(train_length,train_length+validation_length))
# test_range = list(range(train_length+validation_length,train_length+validation_length+test_length))
# all_data_range = list(range(0,all_data_length))
# custom_split_range = list(range(0,custom_split))

# print(train_length)
# print(validation_length)
# print(all_data_length)

# print("Custom_split length", len(custom_split_range))

# train_data_m = torch.utils.data.RandomSampler(train_range,False)
# validation_data_m = torch.utils.data.RandomSampler(val_range,False)
# test_data_m = torch.utils.data.SubsetRandomSampler(test_range,False)
# all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
# custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)

# print("produced dataset split amounts")
# #################################################################################################

# # https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
# print("Full_dataset: ", len(all_data_m))
# print("Training: ", len(train_data_m))
# print("validation: ", len(validation_data_m))

# Train_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
#     sampler=train_data_m)

# Val_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
#     sampler=validation_data_m)

# # Train_data=DataLoader(
# #     dataset=dataset,
# #     batch_size=Param.SegNet.batch_size,
# #     sampler=custom_split_m)

# # Val_data=DataLoader(
# #     dataset=dataset,
# #     batch_size=Param.SegNet.batch_size,
# #     sampler=validation_data_m)
    
    
    
    
    
    ##################################################################################################

print("Loading Dataset")
folder = np.loadtxt(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"] + "/Training_dataset.csv", delimiter=",",dtype=str)

image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

dataset = Load_Dataset(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"],
                       image_folder_in,
                       masks_folder_in,
                       Param.Parameters.Network["Hyperparameters"]["Apply_Augmentation"])

Dataset_size = len(folder)

# split = folder[:,3].astype(int)

# nonempty = folder[:,-1].astype(float)
print("Total number: ", Dataset_size)
print("Non empty masks: ",len(folder[np.where(folder[:,-1].astype(float) > 0)]))
print("Empty masks: ",len(folder[np.where(folder[:,-1].astype(float) == 0)]))
# input("testing outputs")
 
                 
# print(folder[:,-1].astype(float))
# input("")

# training_split = folder[np.where(folder[:,-1].astype(float) > 0),2]

# split here is currently 01 validation (20%) and the rest 23456789 at (80%)
# values are greater than or equal to 3, i.e 3,4,5,6,7,8,9 (70%)
training_split = folder[np.where(folder[:,3].astype(int) >= 3),2]
# training_split = folder[np.where(folder[:,3].astype(int) == 5),2]

# training_split = folder[:,2]
training_split = np.squeeze(training_split).astype(int)

# values are less than 2, i.e 0,1 (20%)
validation_split = folder[np.where(folder[:,3].astype(int) < 2),2]
# validation_split = folder[np.where(folder[:,3].astype(int) == 9),2]
validation_split = np.squeeze(validation_split).astype(int)

# i should split 20% of the dataset off manually to give myself a test set and make it easier to do a val/train split (is this even the right way to go about it?)
# values are equal to 2, i.e 2 (10%)
test_split = folder[np.where(folder[:,3].astype(int) == 2),2]
test_split = np.squeeze(test_split).astype(int)

# train_data_0 = training_split
# random.Random(Param.Parameters.Network["Global"]["Seed"]).shuffle(train_data_0)
# validation_data_0 = validation_split
# random.Random(Param.Parameters.Network["Global"]["Seed"]).shuffle(validation_data_0)

train_data = torch.utils.data.RandomSampler(training_split,False)
validation_data = torch.utils.data.RandomSampler(validation_split,False)

# clearning up CPU space (hopefully) so that the models can be run simulataniously
del folder

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    sampler=train_data, pin_memory=True)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    sampler=validation_data, pin_memory=True)

print("Actual train length", len(Train_data.sampler))
print("actual validation length", len(Val_data.sampler))

train(Train_data, Val_data, load=False)