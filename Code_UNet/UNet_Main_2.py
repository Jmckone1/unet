from Unet_modules.Unet_Main_dataloader import BraTs_Dataset
from Unet_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Unet_modules.Evaluation import DiceLoss
from torch.utils.data import DataLoader
import Net.Unet_components_v2 as net
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
# from os import walk
import numpy as np
import shutil
import torch
import csv
import os

import Unet_modules.Parameters_seg as Param
# import Unet_modules.Dice_Loss as Eval

np.random.seed(Param.Global.Seed)
torch.manual_seed(Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= Param.Global.GPU

np.set_printoptions(precision=4)

print("###################################################")
print("Parameter file values")
print("###################################################")

for val, var in enumerate(vars(Param.Global)):
    if val == len(vars(Param.Global)) - 3:
        print("")
        break
    else:
        print(var, getattr(Param.Global, var))

for val, var in enumerate(vars(Param.sData)):
    if val == len(vars(Param.sData)) - 3:
        print("")
        break
    else:
        print(var, getattr(Param.sData, var))
        
for val, var in enumerate(vars(Param.SegNet)):
    if val == len(vars(Param.SegNet)) - 3:
        print("")
        break
    else:
        print(var, getattr(Param.SegNet, var))

input("Press Enter to continue . . . ")

# criterion = nn.BCEWithLogitsLoss()
criterion = DiceLoss()
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
        truth_input = truth_input.to(Param.SegNet.device)
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.SegNet.device)
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
        # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before validation and calculating final loss values.
        if(truth_input.ndim == 3):
            truth_input = truth_input[np.newaxis,:,:,:]
            label_input = label_input[np.newaxis,:,:]
            
        pred = unet(truth_input)
        pred = pred.squeeze()
        
        if(pred.ndim == 2):
            pred = pred[np.newaxis,:,:]

        loss = criterion(pred, label_input)
        running_loss =+ loss.item()
        #losses.append(running_loss)
        cur_step += 1
        
        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before evaluation.
        if(pred_output.ndim == 2):
            pred_output = pred_output[np.newaxis,:,:]
            truth_output = truth_output[np.newaxis,:,:]
        
        for i in range(cur_batch_size):
            DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))
        
        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + step + "validation_loss.csv", 'a') as f: 
            np.savetxt(f, [running_loss], delimiter=',')
        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + step + "validation_dice.csv", 'a') as f: 
            np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                
    print("Validation complete")
    print(" ")

    # return losses, DS

#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    
    # run UNet.load_weights for loading of frozen or unfrozen models, use UNet for no initialisation.
    # if using UNet.load_weights allow_update = False for Frozen weights, allow_update = True for unfrozen weights
    
    if Param.SegNet.useWeights == True:
        unet = net.UNet.load_weights(Param.SegNet.input_dim, 
                                     Param.SegNet.label_dim, 
                                     Param.SegNet.hidden_dim, 
                                     Param.SegNet.checkpoint_name, 
                                     Param.SegNet.allow_update).to(Param.SegNet.device)
    else:
        unet = net.UNet(Param.SegNet.input_dim, 
                        Param.SegNet.label_dim, 
                        Param.SegNet.hidden_dim).to(Param.SegNet.device)
        
    unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.SegNet.lr, weight_decay=Param.SegNet.weight_decay)
    
    if not os.path.exists("Checkpoints/" + Param.SegNet.c_file + "Model_architecture"):
        os.makedirs("Checkpoints/" + Param.SegNet.c_file + "Model_architecture")
    
    with open("Checkpoints/" + Param.SegNet.c_file + "Model_architecture.txt", 'w') as write: 
        write.write("left_path: " + Param.SegNet.checkpoint_name + "\n")
        write.write("epochs: " + str(Param.SegNet.size) + "\n")
        write.write("batch size: " + str(Param.SegNet.batch_size) + "\n")
        write.write("learning rate: " + str(Param.SegNet.lr) + "\n")
        write.write(str(unet))
        
    original = r'Code_UNet/Unet_modules/Parameters_seg.py'
    target = r'Checkpoints/' + Param.SegNet.c_file + 'Parameters_seg.py'
    shutil.copyfile(original, target)

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #

    for epoch in range(Param.SegNet.n_epochs):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss = 0.0

        for truth_input, label_input in tqdm(Train_data):
            
            DS = []
            
            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(Param.SegNet.device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            label_input = label_input.to(Param.SegNet.device)
            label_input = label_input.float()
            label_input = label_input.squeeze()

            # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before validation and calculating final loss values.
            if(truth_input.ndim == 3):
                truth_input = truth_input[np.newaxis,:,:,:]
                label_input = label_input[np.newaxis,:,:]
                
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()
            
            if(pred.ndim == 2):
                pred = pred[np.newaxis,:,:]
            
            # forward
            unet_loss = criterion(pred, label_input)
            
            # backward
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item()
            # loss_values.append(running_loss)
            cur_step += 1
            
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            
            # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before evaluation.
            if(pred_output.ndim == 2):
                pred_output = pred_output[np.newaxis,:,:]
                truth_output = truth_output[np.newaxis,:,:]
                
            for i in range(cur_batch_size):
                DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))
            
            with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "training_loss.csv", 'a') as f: 
                np.savetxt(f, [running_loss], delimiter=',')
            with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "training_dice.csv", 'a') as f: 
                np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                
#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            # for each display step in the epoch (each 50 by default)
            if cur_step % Param.SegNet.display_step == 0:
                # only for the first epoch and only before step 550
                # this is primarily for the memory concerns of the project, 
                # 550 itself is a arbitrary value chosen becuase of plotting sizes.
                if epoch == 0 and cur_step <= 550:
                    checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                    out = "Checkpoints/" + Param.SegNet.c_file + "checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                    torch.save(checkpoint, out)

                    # if enabled this will validate the model during every *DISPLAY STEP* (default 50 batches) during the first epoch

                    if Param.SegNet.checkpoint_eval == True:
                        if epoch == 0:
                            Validate(unet, criterion, Val_data, epoch, step = "_" + str(cur_step))

        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/" + Param.SegNet.c_file + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)

        Validate(unet, criterion, Val_data, epoch)

    print('Finished Training Dataset')

#####################################################################################################
# dataset length splitting - currently needs testing - the code above is the prior functioning code #
#####################################################################################################
print("starting model")

# apply_transform adds data augmentation to the model - in this case we apply horizontal flip, vertical flip, rotation up to 30 degrees and between 10% and 20% zoom to the center of the image; with 50%, 50%, 25% and 25% chances of occuring.
dataset = BraTs_Dataset(Param.SegNet.dataset_path, path_ext = Param.SegNet.extensions, size=Param.SegNet.size, apply_transform=True)
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

print("Custom_split length", len(custom_split_range))

train_data_m = torch.utils.data.RandomSampler(train_range,False)
validation_data_m = torch.utils.data.RandomSampler(val_range,False)
test_data_m = torch.utils.data.SubsetRandomSampler(test_range,False)
all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)

print("produced dataset split amounts")
#################################################################################################

# https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
print("Full_dataset: ", len(all_data_m))
print("Training: ", len(train_data_m))
print("validation: ", len(validation_data_m))

print("Epochs: ", Param.SegNet.n_epochs)

# Train_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.SegNet.batch_size,
#     sampler=train_data_m)

# Val_data=DataLoader(
#     dataset=dataset,
#     batch_size=Param.SegNet.batch_size,
#     sampler=validation_data_m)

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.SegNet.batch_size,
    sampler=custom_split_m)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.SegNet.batch_size,
    sampler=validation_data_m)

print("Actual train length", len(Train_data.sampler))
print("actual validation length", len(Val_data.sampler))

train(Train_data, Val_data, load=False)