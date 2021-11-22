import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

#from Unet_modules.Brats_dataloader_3 import BraTs_Dataset
#from Unet_modules.Full_model_dataloader_main import BraTs_Dataset
from Unet_modules.Full_model_dataloader_all_data import BraTs_Dataset

import Net.Unet_components_v2 as net
import csv
from os import walk
import nibabel as nib
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from Unet_modules.Evaluation import DiceLoss
from Unet_modules.Evaluation import Dice_Evaluation as Dice_Eval

# In the format "FileName/"
c_file = "Full_model_MK5_H16_unfrozen_5_epochs_DiceLoss/"

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
#criterion = nn.BCEWithLogitsLoss()
criterion = DiceLoss()

n_epochs = 6
input_dim = 4
label_dim = 1
hidden_dim = 16

display_step = 50
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(240 * size)
device = 'cuda'

val_percent = 0.1
test_percent = 0.2
train_percent = 1 - (val_percent + test_percent)

#--------------------------------------------------------#
#              Define validation start                   #

def Validate(unet, criterion, Val_data):
    print(" ")
    print("Validation...")
    unet.eval()
    losses = []
    DS = []
    running_loss = 0.0
    cur_step = 0
    for truth_input, label_input in tqdm(Val_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(device)
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(device)
        label_input = label_input.float()
        label_input = label_input.squeeze()
            
        pred = unet(truth_input)
        pred = pred.squeeze()

        loss = criterion(pred, label_input)
        running_loss =+ loss.item() * truth_input.size(0)
        losses.append(running_loss / len(Val_data))

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        for i in range(cur_batch_size):
            DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))
        #print("Validation Dice Score: ", DS)
        
        cur_step += 1
    metrics = losses
    print("Validation complete")
    print(" ")
    
    return metrics, DS

#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    
    checkpoint_name = "Checkpoints_RANO/Unet_H16_M9_O10A0/checkpoint_99.pth"
    # run UNet.load_weights for loading of frozen or unfrozen models, use UNet for no initialisation.
    # if using UNet.load_weights allow_update = False for Frozen weights, allow_update = True for unfrozen weights
    unet = net.UNet.load_weights(input_dim, label_dim, hidden_dim, checkpoint_name, allow_update=True).to(device)
    
    #unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)
    
    with open("Checkpoints/" + c_file + "Model_architecture", 'w') as write: 
        write.write("left_path: " + checkpoint_name + "\n")
        write.write("epochs: " + str(n_epochs) + "\n")
        write.write("batch size: " + str(batch_size) + "\n")
        write.write("learning rate: " + str(lr) + "\n")
        write.write(str(unet))

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #

    for epoch in range(n_epochs):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss = 0.0
        loss_values = []
        valid_loss = []
        total_loss = []
        DS = []
        
        for truth_input, label_input in tqdm(Train_data):

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
            
            # forward
            unet_loss = criterion(pred, label_input)
            
            # backward
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item() * truth_input.size(0)
            loss_values.append(running_loss / len(Train_data))
            cur_step += 1
            
            # kaggle 2017 2nd place
            # https://www.programcreek.com/python/?project_name=juliandewit%2Fkaggle_ndsb2017
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            for i in range(cur_batch_size):
                DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                #print("Training Dice Score: ", DS)

#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            if cur_step % display_step == 0:

                checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                out = "Checkpoints/" + c_file + "checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                torch.save(checkpoint, out)

#                    Display stage end                   #           
#--------------------------------------------------------#
#               step and loss output start               #

        plt.plot(range(len(loss_values)),loss_values)
        plt.title("Epoch " + str(epoch + 1) + ": loss")

        plt.show()
        total_loss.append(loss_values)
        
        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/" + c_file + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)

        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "training_dice.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(DS)
            
        epoch_val_loss, val_dice = Validate(unet, criterion, Val_data)
        
        valid_loss.append(epoch_val_loss)
        
        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "validation_dice.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(val_dice)
            
        t = []
        v = []
        
        for i in range(len(total_loss)):
            t.append(np.mean(total_loss[i]))
            if len(valid_loss[i]) != 0:
                v.append(np.mean(valid_loss[i]))

        plt.plot(range(len(t)),t)
        plt.plot(range(len(v)),v)
        plt.legend(["training","validation"])
        plt.show()

    print('Finished Training Dataset')
    return total_loss, valid_loss

#               step and loss output start               #
#--------------------------------------------------------#

dataset = BraTs_Dataset("Brats_2018_data_all/All_data",path_ext = ["/HGG","/LGG"],size=size,apply_transform=True)
Validation_dataset = BraTs_Dataset("Brats_2018_data_split/Validation", path_ext=["/HGG","/LGG"],size=size,apply_transform=True)

print("Training: ", len(dataset))
print("validation: ", len(Validation_dataset))

Train_data = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True)

Val_data = DataLoader(
    dataset=Validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True)

Train_loss,validation_loss = train(Train_data, Val_data, load=False)