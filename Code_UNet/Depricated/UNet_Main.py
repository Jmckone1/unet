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
import torch
import csv
import os

import shutil

import Unet_modules.Parameters_seg as Param

np.random.seed(Param.Global.Seed)
torch.manual_seed(Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= Param.Global.GPU

np.set_printoptions(precision=4)

print("###################################################")
print("Parameter file values")
print("###################################################")
print("Current Seed value",Param.Global.Seed)
print("Current GPU allocated",Param.Global.GPU)
print("Dataloader input path, IMAGE",Param.sData.image_in)
print("Dataloader input path, RANO",Param.sData.rano_in)
print("Dataloader index path",Param.sData.index_file)
print("Dataset path",Param.SegNet.dataset_path)
print("Checkpoint save path",Param.SegNet.c_file)
print("Encode checkpoint load path",Param.SegNet.checkpoint_name)
print("Total epochs",Param.SegNet.n_epochs)
print("Input layer dimensions",Param.SegNet.input_dim)
print("Label dimensions",Param.SegNet.label_dim)
print("Hidden layer dimensions",Param.SegNet.hidden_dim)
print("Current Learning rate",Param.SegNet.lr)
print("Current interpolation size",Param.SegNet.size)
print("Current console output step",Param.SegNet.display_step)
print("Current batch size",Param.SegNet.batch_size)
print("Device",Param.SegNet.device)
print("Training set split percent",Param.SegNet.train_split)
print("Validation set split percent",Param.SegNet.validation_split)
print("testing set split percent",Param.SegNet.test_split)
print("Custom splitting percent",Param.SegNet.custom_split_amount)
print("Weight decay",Param.SegNet.weight_decay)
print("File path extension",Param.SegNet.extensions)
print("Apply encoder weights",Param.SegNet.useWeights)
print("Allow encoder update",Param.SegNet.allow_update)
print("###################################################")
input("Press Enter to continue . . . ")

criterion = nn.BCEWithLogitsLoss()

#update file read in counter based on the UNet_RANO_cosine code
#update dataloader based on the code_UNet_RANO dataloaders

#--------------------------------------------------------#
#              Define validation start                   #

# for some reason this started taking up to 10x as long per batch to run validation over training - not really ure why, cleaned it up as much as i can but. . . . . . . . . . . .
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
        truth_input = truth_input.to(Param.SegNet.device)
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.SegNet.device)
        label_input = label_input.float()
        label_input = label_input.squeeze()
            
        pred = unet(truth_input)
        pred = pred.squeeze()

        loss = criterion(pred, label_input)
        running_loss =+ loss.item()
        losses.append(running_loss)
        cur_step += 1
        
        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        for i in range(cur_batch_size):
            DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))

    print("Validation complete")
    print(" ")

    return losses, DS

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
        loss_values = []
        valid_loss = []
        total_loss = []
        DS = []
        
        for truth_input, label_input in tqdm(Train_data):

            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(Param.SegNet.device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            label_input = label_input.to(Param.SegNet.device)
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

            running_loss =+ unet_loss.item()
            loss_values.append(running_loss)
            cur_step += 1
            
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            for i in range(cur_batch_size):
                DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))

#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            if cur_step % Param.SegNet.display_step == 0:

                checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                out = "Checkpoints/" + Param.SegNet.c_file + "checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                torch.save(checkpoint, out)
                
                # save the loss and dice score - then run validation and save the same for each of the display 
                # checkpoint values in the defined epoch (this case epoch 0)
                
                # This should also serve to check if the validation code is working before we get the end of the epoch
                checkpoint_eval = True # this will need to be moved to the param file if functional
                if checkpoint_eval == True:
                    if epoch == 0:

                        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" 
                                  + str(epoch) + "_" + str(cur_step) +  "_" + "training_loss.csv", 'w') as f: 
                            write = csv.writer(f) 
                            write.writerow(loss_values)

                        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" 
                                  + str(epoch) + "_" + str(cur_step) +  "_" + "training_dice.csv", 'w') as f: 
                            write = csv.writer(f) 
                            write.writerow(DS)

                        epoch_val_loss, val_dice = Validate(unet, criterion, Val_data)
                        valid_loss.append(epoch_val_loss)

                        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" 
                                  + str(epoch) + "_" + str(cur_step) +  "_"+ "validation_loss.csv", 'w') as f: 
                            write = csv.writer(f) 
                            write.writerow(valid_loss)

                        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" 
                                  + str(epoch) + "_" + str(cur_step) +  "_"+ "validation_dice.csv", 'w') as f: 
                            write = csv.writer(f) 
                            write.writerow(val_dice)
                        epoch_val_loss, val_dice
#                    Display stage end                   #           
#--------------------------------------------------------#
#               step and loss output start               #

        plt.plot(range(len(loss_values)),loss_values)
        plt.title("Epoch " + str(epoch + 1) + ": loss")

        plt.show()
        total_loss.append(loss_values)
        
        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/" + Param.SegNet.c_file + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)
        
        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "training_dice.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(DS)
            
        epoch_val_loss, val_dice = Validate(unet, criterion, Val_data)
        
        valid_loss.append(epoch_val_loss)

        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
        with open("Checkpoints/" + Param.SegNet.c_file + "epoch_" + str(epoch) + "validation_dice.csv", 'w') as f: 
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

# split_amount = 1

# data_size = len(dataset)
# patients_number = data_size / 155 # this need to be changed to not include a hard coded 155

# train_length = int(155*(np.ceil(patients_number * train_split)))
# validation_length = int(155*(np.floor(patients_number * validation_split)))
# test_length = int(155*(np.floor(patients_number * test_split)))

# split_1 = list(range(0,int(155*(np.ceil((train_length / 155) * split_amount)))))

# train_range = list(range(0,train_length))
# val_range = list(range(train_length,train_length+validation_length))

# train_data_m = torch.utils.data.RandomSampler(train_range,False)
# validation_data_m = torch.utils.data.RandomSampler(val_range,False)

# data_split_m = torch.utils.data.RandomSampler(split_1,False)

# print("Training: ", len(train_data_m))
# print("Actual_input: ", len(split_1))f
# print("validation: ", len(validation_data_m))

# Train_data=DataLoader(
#     dataset=dataset,
#     batch_size=batch_size,
#     sampler=split_1)

# Val_data=DataLoader(
#     dataset=dataset,
#     batch_size=batch_size,
#     sampler=validation_data_m)

##################################################################################################################################
# dataset length splitting - currently needs testing - the code above is the prior functioning code ##############################
##################################################################################################################################
print("starting model")
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

train_data_m = torch.utils.data.RandomSampler(train_range,False)
validation_data_m = torch.utils.data.RandomSampler(val_range,False)
test_data_m = torch.utils.data.SubsetRandomSampler(test_range,False)
all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)

print("produced dataset split amounts")
##################################################################################################################################

# https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
print("Full_dataset: ", len(all_data_m))
print("Training: ", len(train_data_m))
print("validation: ", len(validation_data_m))

print("Epochs: ", Param.SegNet.n_epochs)

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.SegNet.batch_size,
    sampler=train_data_m)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.SegNet.batch_size,
    sampler=validation_data_m)

Train_loss,validation_loss = train(Train_data, Val_data, load=False)