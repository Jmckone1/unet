####################################################################################################
# The primary folder used for runngin the networks aafter the 04/2022 dataset normailsation rework #
# Unet running RANO regression, Splitting the dataset into train, val and test automatically while #
# utilising some memory improvement functions hence the file naming conventions.                   #
####################################################################################################

from Unet_modules.RANO_dataloader_2_scandir import BraTs_Dataset
from Unet_modules.Evaluation import Jaccard_Evaluation as Jacc
from Unet_modules.Penalty_3 import Penalty
from sklearn.metrics import jaccard_score

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# from torch import nn # need this to use built in loss functions

import Net.Unet_Rano_components as net
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import numpy as np
import logging
import shutil
import torch
import csv
import os

import Unet_modules.Parameters as Param

# np.random.seed(Param.Global.Seed)
# torch.manual_seed(Param.Global.Seed)

np.random.seed(Param.Global.Seed)
torch.manual_seed(Param.Global.Seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=Param.Global.GPU

print(torch.__version__)
print("")

# In the format "FileName/"
# c_file = "Unet_H16_M13_O10/"
# model 11 makes use of the new orthogonality penalty at 50 epochs
# model 12 makes use of the new orthogonality penalty at 50 epochs including the new quality dataset
# (although there is an issue with the ground truth dataset being incorrect for the dataset in this case).
# model 13 makes use of the new updated ground truth so make sure you get it right this time
# going to have a rethink on the naming conventions used for the models here - currently rerunning the dataset aquisition code for the rano part of the model.
# unet_h16_m13_o0_2 is reran with default as i think it was mimicing the results from the orth 100 penalty
# unet_h16_m13_o100_2 is reran using the orginal penalty measure for comparison with orth at 100x penalty
# these models prodcued no meaningful difference in results - good from an overall point of view with up to 0.4 jaccard after 50 eposchs whihc is up from the previous, but no change at all in jaccard otherwise which is worrying

# models with cosine similarity loss now (18052022) which im printing out the scores for each of the loss measures (MSE = (COSINE + w))

np.set_printoptions(precision=4)

print("###################################################")
print("Parameter file values")
print("###################################################")
print("Current Seed value", Param.Global.Seed)
print("Device", Param.rNet.device)
print("Input Dimension", Param.rNet.input_dim)
print("Label Dimension", Param.rNet.label_dim)
print("Hidden Layer Dimension", Param.rNet.hidden_dim)
print("Checkpoint Path", Param.rNet.checkpoint)
print("Epoch total number", Param.rNet.n_epochs)
print("Batch Size number", Param.rNet.batch_size)
print("Learning Rate", Param.rNet.lr)
print("Orthogonality Penalty value", Param.rNet.orth_penalty)
# print("Area Penalty value", Param.rNet.area_penalty)
print("Console Display step", Param.rNet.display_step)
print("Dataset path", Param.rNet.dataset_path)
print("Interpolation multiplier", Param.rNet.size)
print("Index counter filepath", Param.rData.index_file)
print("Training Split value", Param.rNet.train_split)
print("Validation Split value", Param.rNet.validation_split)
print("Testing Split value", Param.rNet.test_split)
print("Custom Split value", Param.rNet.custom_split_amount)
print("###################################################")
input("Press Enter to continue . . . ")

#criterion = nn.MSELoss()
loss_f = Penalty(Param.rNet.orth_penalty)#, Param.rNet.area_penalty)
criterion = loss_f.MSELossorthog

### added a bunch of outputs for here to test when i get back!!!
def Validate(unet, criterion, Val_data):
    print(" ")
    print("Validation...")
    unet.eval()
    
    mse_values = []
    cosine_values = []
    losses = []
    
    running_loss = 0.0
    mse_run = 0.0
    cosine_run = 0.0
    
    cur_step = 0
    jaccard_val = []
    
    for truth_input, label_input in tqdm(Val_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.rNet.device)
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.rNet.device)
        label_input = label_input.float()
        label_input = label_input.squeeze()

        pred = unet(truth_input)
        pred = pred.squeeze()

        # forward
        loss, mse, cosine = criterion(pred, label_input)
        # print("v loss", loss)
        
        loss.backward()
        
        running_loss =+ loss.item()
        mse_run =+ mse.item()
        cosine_run =+ cosine.item()
        # print("v run loss", running_loss)
        
        losses.append(running_loss)
        mse_values.append(mse_run)
        cosine_values.append(cosine_run)
        
        # print("v losses", losses)

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        for input_val in range(cur_batch_size):
            
            corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
            mask_truth = Jacc.mask((240,240),corners_truth)#*1
            corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
            mask_pred = Jacc.mask((240,240),corners_pred)#*1
            
            # print("Total sum of mask pixels", np.sum(np.sum(mask_truth)))
            if np.sum(np.sum(mask_truth)) > 2:
                jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
            else:
                jaccard_val.append(float("NaN"))
        
        cur_step += 1
    
    print("v j val",jaccard_val)
    print("Validation complete")
    print(" ")
    
    return losses, mse_values, cosine_values, jaccard_val

#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    Improvement = 0
    
    unet = net.UNet(Param.rNet.input_dim, Param.rNet.label_dim, Param.rNet.hidden_dim).to(Param.rNet.device)
    print(unet)
    
    if not os.path.exists("Checkpoints_RANO/" + Param.rNet.checkpoint):
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint)
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss_mse")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss_cosine")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss_mse")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss_cosine")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_Jaccard")
        os.makedirs("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_Jaccard")
    
    ## may get rid of this section in the future as we now save the parameter file, though the architecture is potentially useful.
    with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Model_architecture", 'w') as write: 
        write.write("epochs: " + str(Param.rNet.n_epochs) + "\n")
        write.write("batch size: " + str(Param.rNet.batch_size) + "\n")
        write.write("learning rate: " + str(Param.rNet.lr) + "\n")
        write.write("orthogonality weight: " + str(Param.rNet.orth_penalty) + "\n")
        write.write("area weight: " + str(Param.rNet.area_penalty) + "\n")

        write.write(str(unet))
        
    ################################################################################################################
    # saves a copy of the current parameter file to the checkpoint file for future reference and reproducability
    # Implemented 11/05/2022 for code files H16_M13 and above
    ################################################################################################################

    original = r'Code_UNet/Unet_modules/Parameters.py'
    target = r'Checkpoints_RANO/' + Param.rNet.checkpoint + 'Parameters.py'
    shutil.copyfile(original, target)

    ################################################################################################################

    unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.rNet.lr, betas=Param.rNet.Betas, weight_decay=Param.rNet.Weight_Decay)

    if load == True:
        checkpoint = torch.load("Checkpoints_RANO/" + Param.rNet.checkpoint + "checkpoint_0_step_1900.pth")

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #
    t = []
    v = []
    total_loss = []
    
    scaler = amp.GradScaler(enabled = True)

    for epoch in range(Param.rNet.n_epochs):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss = 0.0
        mse_run = 0.0
        cosine_run = 0.0
        
        loss_values = []
        mse_values = []
        cosine_values = []
        
        valid_loss = []
        valid_mse_values = []
        valid_cosine_values = []
        
        jaccard = []
        valid_jaccard = []
        
        for truth_input, label_input in tqdm(Train_data):

            cur_batch_size = len(truth_input)
            
#             print(truth_input.float().type())
#             print(label_input.float().type())
#             input(" ")
            
            # flatten ground truth and label masks
            truth_input = truth_input.to(Param.rNet.device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            label_input = label_input.to(Param.rNet.device)
            label_input = label_input.float()
            label_input = label_input.squeeze()

            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()

            # forward
            unet_loss, mse, cosine = criterion(pred, label_input)
            
            for input_val in range(cur_batch_size):
                
                corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
                mask_truth = Jacc.mask((240,240),corners_truth)*1
                corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
                mask_pred = Jacc.mask((240,240),corners_pred)*1
                
                if np.sum(np.sum(mask_pred)) > 2:
                    jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    jaccard.append(float("NaN"))
            
            # backward
            scaler.scale(unet_loss).backward()
            scaler.step(unet_opt)
            scaler.update()

            running_loss =+ unet_loss.item()
            
            mse_run =+ mse.item()
            cosine_run =+ cosine.item() 
            
            # removed the * truth_input.size(0) from all examples of the .item() 
            # which would multiply the values by the batch size, not really needed in this case, 
            # i can do this outside if necessary
            
#             print(cosine.item())
#             print(truth_input.size(0))
#             print(len(Train_data))
#             input("")
            
            cur_step += 1

#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

#             loss_values.append(running_loss / len(Train_data))
#             total_loss.append(loss_values)
        
#             mse_values.append(mse_run/len(Train_data))
#             cosine_values.append(cosine_run/len(Train_data))

            loss_values.append(running_loss)
            mse_values.append(mse_run)
            cosine_values.append(cosine_run)
            total_loss.append(loss_values)
        
            if cur_step % Param.rNet.display_step == 0:

                print("Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                print(label_input[0,:].shape)
                
                # this section of code doesnt work due to the 4 channel indexing for the ssingle channel data that is out of range, wasnt really being sued anyways
                
                # Print jaccard for current output in the batch
#                 print("index", jaccard[-cur_batch_size:]) 
#                 print("")
                    
#                 for i in range(cur_batch_size):
                    
# #                     print("input", label_input[i,:].data.cpu().numpy())
# #                     print("prediction",pred[i,:].data.cpu().numpy())
                    
#                     f, axarr = plt.subplots(1,2)

#                     data_in = label_input[i,:].data.cpu().numpy()
#                     D1 = np.asarray([[data_in[1],data_in[3]],[data_in[0],data_in[2]]]) 
#                     D2 = np.asarray([[data_in[5],data_in[7]],[data_in[4],data_in[6]]]) 
                    
#                     axarr[0].imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')
#                     axarr[0].plot(D1[0, :], D1[1, :], lw=2, c='r')
#                     axarr[0].plot(D2[0, :], D2[1, :], lw=2, c='b')
#                     axarr[0].set_title('Truth')
                    
#                     data_out = pred[i,:].data.cpu().numpy()
#                     D1 = np.asarray([[data_out[1],data_out[3]],[data_out[0],data_out[2]]]) 
#                     D2 = np.asarray([[data_out[5],data_out[7]],[data_out[4],data_out[6]]]) 

#                     axarr[1].imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')
#                     axarr[1].plot(D1[0, :], D1[1, :], lw=2, c='r')
#                     axarr[1].plot(D2[0, :], D2[1, :], lw=2, c='b')
#                     axarr[1].set_title('Prediction')
                    
#                     plt.show()
                   
                # kaggle 2017 2nd place
                # https://www.programcreek.com/python/?project_name=juliandewit%2Fkaggle_ndsb2017
                pred_output = pred.cpu().detach().numpy()
                truth_output = label_input.cpu().detach().numpy()
                
                plt.plot(range(len(loss_values)),loss_values)
                plt.title("Epoch " + str(epoch + 1) + ": loss")

                plt.show()

#                    Display stage end                   #           
#--------------------------------------------------------#
#               step and loss output start   #
        epoch_val_loss, epoch_valid_mse, epoch_valid_cosine, epoch_jaccard_valid = Validate(unet, criterion, Val_data)
        
        valid_loss.append(epoch_val_loss)
        valid_mse_values.append(epoch_valid_mse)
        valid_cosine_values.append(epoch_valid_cosine)
        
        valid_jaccard.append(epoch_jaccard_valid)

        print("Improvement", Improvement)
        print("nan mean jaccard validation over the epoch", np.nanmean(epoch_jaccard_valid))
        print("mean jaccard over epoch with nan", epoch_jaccard_valid)
        print("")
              
        # save a checkpoint only if there has been an improvement in the total jaccard score for the model.
        if np.nanmean(epoch_jaccard_valid) > Improvement:
            if np.nanmean(epoch_jaccard_valid) == np.isnan:
                Improvement = 0
            else:
                Improvement = np.nanmean(epoch_jaccard_valid)
        
            print("saving epoch: ", epoch)
            checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
            out = "Checkpoints_RANO/" + Param.rNet.checkpoint + "checkpoint_" + str(epoch) + ".pth"
            torch.save(checkpoint, out)
            
        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss/epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss_mse/epoch_" + str(epoch) + "training_loss_mse.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(mse_values)
            
        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss_cosine/epoch_" + str(epoch) + "training_loss_cosine.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(cosine_values)

        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss/epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss_mse/epoch_" + str(epoch) + "validation_loss_mse.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_mse_values)
            
        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss_cosine/epoch_" + str(epoch) + "validation_loss_cosine.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_cosine_values)

        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_Jaccard/epoch_" + str(epoch) + "jaccard_index.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(jaccard)

        with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_Jaccard/epoch_" + str(epoch) + "validation_jaccard_index.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_jaccard)

    print('Finished Training Dataset')
    return total_loss, valid_loss

#               step and loss output start               #
#--------------------------------------------------------#

dataset = BraTs_Dataset(Param.rNet.dataset_path, path_ext = Param.rNet.Extensions, size=Param.rNet.size, apply_transform=False)
##################################################################################################################################
# dataset length splitting - currently needs testing - the code below is the prior functioning code ##############################
##################################################################################################################################
index_f = np.load(Param.rNet.dataset_path + Param.rData.index_file)
patients_number = len(index_f)

train_length = index_f[int(np.floor(patients_number*Param.rNet.train_split))]
validation_length = index_f[int(np.ceil(patients_number*Param.rNet.validation_split))]
test_length = index_f[int(np.ceil(patients_number*Param.rNet.test_split))-1]
all_data_length = index_f[-1]
custom_split = index_f[int(np.ceil(patients_number*Param.rNet.custom_split_amount))-1]

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
##################################################################################################################################

# https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
print("Full_dataset: ", len(all_data_m))
print("Training: ", len(train_data_m))
print("validation: ", len(validation_data_m))

print("Epochs: ", Param.rNet.n_epochs)
print("Orthogonality Penalty:", Param.rNet.orth_penalty)
print("Area Penalty: ", Param.rNet.area_penalty)

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.rNet.batch_size,
    sampler=train_data_m)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.rNet.batch_size,
    sampler=validation_data_m)

Train_loss, validation_loss = train(Train_data, Val_data, load=False)