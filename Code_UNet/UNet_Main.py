from Unet_modules.Full_model_dataloader_all_data import BraTs_Dataset
from Unet_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Unet_modules.Evaluation import DiceLoss
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import Net.Unet_components_v2 as net
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
from os import walk
import numpy as np
import torch
import csv
import os

import Unet_modules.Parameters_seg as Param

np.random.seed(Param.Global.Seed)
torch.manual_seed(Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= Param.Global.GPU

c_file = Param.SegNet.c_file
size = Param.SegNet.size
n_epochs = Param.SegNet.n_epochs
input_dim = Param.SegNet.input_dim
label_dim = Param.SegNet.label_dim
hidden_dim = Param.SegNet.hidden_dim
display_step = Param.SegNet.display_step
batch_size = Param.SegNet.batch_size
lr = Param.SegNet.lr
device = Param.SegNet.device

train_split = Param.SegNet.train_split
validation_split = Param.SegNet.validation_split
test_split = Param.SegNet.test_split

weight_decay= Param.SegNet.weight_decay

checkpoint_name = Param.SegNet.checkpoint_name

criterion = nn.BCEWithLogitsLoss()

#update file read in counter based on the UNet_RANO_cosine code
#update dataloader based on the code_UNet_RANO dataloaders

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
    
    # run UNet.load_weights for loading of frozen or unfrozen models, use UNet for no initialisation.
    # if using UNet.load_weights allow_update = False for Frozen weights, allow_update = True for unfrozen weights
    unet = net.UNet.load_weights(input_dim, label_dim, hidden_dim, checkpoint_name, allow_update=True).to(device)
    
    # unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
    
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
            
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            for i in range(cur_batch_size):
                DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))

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
        
        epoch_val_loss, val_dice = Validate(unet, criterion, Val_data)
        valid_loss.append(epoch_val_loss)

        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        with open("Checkpoints/" + c_file + "epoch_" + str(epoch) + "training_dice.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(DS)
        
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

dataset = BraTs_Dataset("Brats_2018_data/Brats_2018_data",path_ext = ["/HGG","/LGG"],size=size,apply_transform=True)

split_amount = 1

data_size = len(dataset)
patients_number = data_size / 155

train_length = int(155*(np.ceil(patients_number * train_split)))
validation_length = int(155*(np.floor(patients_number * validation_split)))
test_length = int(155*(np.floor(patients_number * test_split)))

split_1 = list(range(0,int(155*(np.ceil((train_length / 155) * split_amount)))))

train_range = list(range(0,train_length))
val_range = list(range(train_length,train_length+validation_length))

train_data_m = torch.utils.data.RandomSampler(train_range,False)
validation_data_m = torch.utils.data.RandomSampler(val_range,False)

data_split_m = torch.utils.data.RandomSampler(split_1,False)

print("Training: ", len(train_data_m))
print("Actual_input: ", len(split_1))
print("validation: ", len(validation_data_m))

Train_data=DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=split_1)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=validation_data_m)

Train_loss,validation_loss = train(Train_data, Val_data, load=False)