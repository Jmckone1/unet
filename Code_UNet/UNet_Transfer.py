import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

#from Unet_modules.Brats_dataloader_3 import BraTs_Dataset
from Unet_modules.Full_model_dataloader import BraTs_Dataset

from Unet_modules.dataloader_test import Test_Dataset
import Net.Full_UNet_components as net
import csv
from os import walk
import nibabel as nib
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

val_percent = 0.1
test_percent = 0.2
train_percent = 1 - (val_percent + test_percent)

# https://nipy.org/nibabel/gettingstarted.html

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

#--------------------------------------------------------#
#             show output tensors start                  #

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28),title=""):

    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.title(title)
    plt.imshow((image_grid.permute(1, 2, 0).squeeze()* 255).type(torch.uint8))
    plt.show()

#              show output tensors end                   #
#--------------------------------------------------------#
#              Define validation start                   #

def Validate(unet, criterion, Val_data):
    print(" ")
    print("Validation...")
    unet.eval()
    losses = []
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
        DS = []
        for i in range(cur_batch_size):
            DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
        print("Validation Dice Score: ", DS)
        
        cur_step += 1
    metrics = losses
    print("Validation complete")
    print(" ")
    
    return metrics


#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    
    unet = net.UNet(input_dim, label_dim, hidden_dim, "Checkpoints_RANO/checkpoint_0.pth").to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

    if load == True:
        checkpoint = torch.load("Checkpoints/checkpoint_0_step_1900.pth")

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

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
            
            cur_step += 1

#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            if cur_step % 250 == 0:
                checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                out = "Checkpoints/checkpoint_" + str(epoch) + "_step_" + str(cur_step) + ".pth"
                torch.save(checkpoint, out)

            if cur_step % display_step == 0:

                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                show_tensor_images(truth_input[:,1,:,:], size=(label_dim, target_shape, target_shape),
                                   title="Flair Input Channel ( channel 2 of 4 )")
                show_tensor_images(label_input, size=(label_dim, target_shape, target_shape),title="Real Labels")
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape),title="Predicted Output")
                plt.plot(range(len(loss_values)),loss_values)
                plt.show()

                # I can make this into a function
                # kaggle 2017 2nd place
                # https://www.programcreek.com/python/?project_name=juliandewit%2Fkaggle_ndsb2017
                pred_output = pred.cpu().detach().numpy()
                truth_output = label_input.cpu().detach().numpy()
                DS = []
                for i in range(cur_batch_size):
                    DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                print("Training Dice Score: ", DS)

#                    Display stage end                   #           
#--------------------------------------------------------#
#               step and loss output start               #

        loss_values.append(running_loss / len(Train_data))
  
        plt.plot(range(len(loss_values)),loss_values)
        plt.title("Epoch " + str(epoch + 1) + ": loss")

        plt.show()
        total_loss.append(loss_values)
        
        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)
        
        with open("Checkpoints/epoch_" + str(epoch) + "training_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)

        valid_loss.append(Validate(unet, criterion, Val_data))

        with open("Checkpoints/epoch_" + str(epoch) + "validation_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
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

dataset = BraTs_Dataset("Brats_2018_data_split/Training",path_ext = ["/HGG","/LGG"],size=size,apply_transform=True)

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