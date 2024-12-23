####################################################################################################
# The primary folder used for runngin the networks aafter the 04/2022 dataset normailsation rework #
# Unet running RANO regression, Splitting the dataset into train, val and test automatically while #
# utilising some memory improvement functions hence the file naming conventions.                   #
####################################################################################################

from Unet_modules.RANO_dataloader_2_scandir import BraTs_Dataset
from Unet_modules.Evaluation import Jaccard_Evaluation as Jacc
from Unet_modules.Penalty_2 import Penalty
from sklearn.metrics import jaccard_score

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# from torch import nn # need this to use built in loss functions

import Net.Unet_Rano_components as net
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import numpy as np
import logging
import torch
import csv
import os

np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# In the format "FileName/"
c_file = "Unet_H16_M12_O100/"
# model 11 makes use of the new orthogonality penalty at 50 epochs
# model 12 makes use of the new orthogonality penalty at 50 epochs including the new quality dataset

# going to have a rethink on the naming conventions used for the models here - currently rerunning the dataset aquisition code for the rano part of the model.

np.set_printoptions(precision=4)

# image interpolation multiplier
# this does not work at this time for the RANO implementation
size = 1

# inital testing showed 50 as the best estimated region before plautau though this may change.
n_epochs = 250
input_dim = 4
label_dim = 8
hidden_dim = 16
orth_penalty = 100
area_penalty = 0 
# area penalty value is currently redundant and will not produce any impact for the penalty 2 model as it has not been implemented - this is purposeful until the point in time where we can test if there is any reasonable point or evidence in it working.

#criterion = nn.MSELoss()
loss_f = Penalty(orth_penalty,area_penalty)
criterion = loss_f.MSELossorthog

display_step = 200
batch_size = 16
lr = 0.0001
initial_shape = int(240 * size)
target_shape = int(8)
device = 'cuda'

###########################################################################################
# ATTEMPT 1 AT LOGGING ERRORS (didnt work but going to leave for the time being)          #
###########################################################################################
# # Create a logging instance
# logger = logging.getLogger('my_application')
# logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR

# # Assign a file-handler to that instance
# fh = logging.FileHandler("ERROR_UNET_1.txt")
# fh.setLevel(logging.INFO) # again, you can set this differently

# # Format your logs (optional)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter) # This will set the format to the file handler

# # Add the handler to your logging instance
# logger.addHandler(fh)

# try:
#     raise ValueError("Some error occurred")
# except ValueError as e:
#     logger.exception(e) # Will send the errors to the file
###########################################################################################

def Validate(unet, criterion, Val_data):
    print(" ")
    print("Validation...")
    unet.eval()
    losses = []
    running_loss = 0.0
    cur_step = 0
    jaccard_val = []
    
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
        
        loss.backward()
        
        running_loss =+ loss.item() * truth_input.size(0)
        losses.append(running_loss / len(Val_data))

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        for input_val in range(cur_batch_size):
                
            corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
            mask_truth = Jacc.mask((240,240),corners_truth)*1
            corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
            mask_pred = Jacc.mask((240,240),corners_pred)*1

            if np.sum(np.sum(mask_pred)) > 2:
                jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
            else:
                jaccard_val.append(float("NaN"))
        
        cur_step += 1
        
    print("Validation complete")
    print(" ")
    
    return losses, jaccard_val

#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

# def train(Train_data,Val_data,load=False):
#     Improvement = 0
    
    unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
    print(unet)
    
#     if not os.path.exists("Checkpoints_RANO/" + c_file):
#         os.makedirs("Checkpoints_RANO/" + c_file)
#         os.makedirs("Checkpoints_RANO/" + c_file + "Training_loss")
#         os.makedirs("Checkpoints_RANO/" + c_file + "Validation_loss")
#         os.makedirs("Checkpoints_RANO/" + c_file + "Training_Jaccard")
#         os.makedirs("Checkpoints_RANO/" + c_file + "Validation_Jaccard")
    
#     with open("Checkpoints_RANO/" + c_file + "Model_architecture", 'w') as write: 
#         write.write("epochs: " + str(n_epochs) + "\n")
#         write.write("batch size: " + str(batch_size) + "\n")
#         write.write("learning rate: " + str(lr) + "\n")
#         write.write("orthogonality weight: " + str(orth_penalty) + "\n")
#         write.write("area weight: " + str(area_penalty) + "\n")

#         write.write(str(unet))
    
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr,betas=(0.9, 0.999), weight_decay=1e-8)

    if load == True:
        checkpoint = torch.load("Checkpoints_RANO/" + c_file + "checkpoint_0_step_1900.pth")

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

# #                   Define model end                     #
# #--------------------------------------------------------#
# #                   Run model start                      #
#     t = []
#     v = []
    
#     scaler = amp.GradScaler(enabled = True)

#     for epoch in range(n_epochs):
#         cur_step = 0
        
#         print("Training...")
#         if epoch == 0 and load == True:
#             epoch = checkpoint['epoch'] + 1
            
#         unet.train()
        
#         running_loss = 0.0
#         loss_values = []
#         valid_loss = []
#         total_loss = []
#         jaccard = []
#         valid_jaccard = []
        
#         for truth_input, label_input in tqdm(Train_data):

#             cur_batch_size = len(truth_input)

#             # flatten ground truth and label masks
#             truth_input = truth_input.to(device)
#             truth_input = truth_input.float() 
#             truth_input = truth_input.squeeze()
#             label_input = label_input.to(device)
#             label_input = label_input.float()
#             label_input = label_input.squeeze()
            
#             image = truth_input.to(device)
#             image = image.float()
#             image = image.squeeze()
            
#             image.requires_grad_()
            
#             # set accumilated gradients to 0 for param update
#             unet_opt.zero_grad()
#             with amp.autocast(enabled = True):
#                 pred = unet(truth_input)
#                 pred = pred.squeeze()

#             # forward
#             unet_loss = criterion(pred, label_input)
            
#             for input_val in range(cur_batch_size):
                
#                 corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
#                 mask_truth = Jacc.mask((240,240),corners_truth)*1
#                 corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
#                 mask_pred = Jacc.mask((240,240),corners_pred)*1
                
#                 if np.sum(np.sum(mask_pred)) > 2:
#                     jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
#                 else:
#                     jaccard.append(float("NaN"))
            
#             # backward
#             scaler.scale(unet_loss).backward()
#             scaler.step(unet_opt)
#             scaler.update()

#             running_loss =+ unet_loss.item() * truth_input.size(0)
            
#             cur_step += 1

# #                     Run model end                      #
# #--------------------------------------------------------#         
# #                  Display stage start                   #

#             loss_values.append(running_loss / len(Train_data))
#             total_loss.append(loss_values)
        
#             if cur_step % display_step == 0:

#                 print("Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
#                 print(label_input[0,:].shape)
                
#                 # Print jaccard for current output in the batch
#                 print("index", jaccard[-cur_batch_size:]) 
#                 print("")
                    
#                 for i in range(cur_batch_size):
                    
#                     print("input", label_input[i,:].data.cpu().numpy())
#                     print("prediction",pred[i,:].data.cpu().numpy())
                    
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
                   
#                 # kaggle 2017 2nd place
#                 # https://www.programcreek.com/python/?project_name=juliandewit%2Fkaggle_ndsb2017
#                 pred_output = pred.cpu().detach().numpy()
#                 truth_output = label_input.cpu().detach().numpy()
                
#                 plt.plot(range(len(loss_values)),loss_values)
#                 plt.title("Epoch " + str(epoch + 1) + ": loss")

#                 plt.show()

# #                    Display stage end                   #           
# #--------------------------------------------------------#
# #               step and loss output start   #
#         epoch_val_loss, epoch_jaccard_valid = Validate(unet, criterion, Val_data)
        
#         valid_loss.append(epoch_val_loss)
#         valid_jaccard.append(epoch_jaccard_valid)
      
#  ####################################################################################################################

#         print(Improvement)
#         print(np.nanmean(epoch_jaccard_valid))
#         print(epoch_jaccard_valid)
#         print("")
              
#         # save a checkpoint only if there has been an improvement in the total jaccard score for the model.
#         if np.nanmean(epoch_jaccard_valid) > Improvement:
#             if np.nanmean(epoch_jaccard_valid) == np.isnan:
#                 Improvement = 0
#             else:
#                 Improvement = np.nanmean(epoch_jaccard_valid)
        
#             print("saving epoch: ", epoch)
#             checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
#             out = "Checkpoints_RANO/" + c_file + "checkpoint_" + str(epoch) + ".pth"
#             torch.save(checkpoint, out)
            
#         with open("Checkpoints_RANO/" + c_file + "Training_loss/epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
#             write = csv.writer(f) 
#             write.writerow(loss_values)

#         with open("Checkpoints_RANO/" + c_file + "Validation_loss/epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
#             write = csv.writer(f) 
#             write.writerow(valid_loss)

#         with open("Checkpoints_RANO/" + c_file + "Training_Jaccard/epoch_" + str(epoch) + "jaccard_index.csv", 'w') as f: 
#             write = csv.writer(f) 
#             write.writerow(jaccard)

#         with open("Checkpoints_RANO/" + c_file + "Validation_Jaccard/epoch_" + str(epoch) + "validation_jaccard_index.csv", 'w') as f: 
#             write = csv.writer(f) 
#             write.writerow(valid_jaccard)
#             #################################################################################################################################

#     print('Finished Training Dataset')
#     return total_loss, valid_loss

# #               step and loss output start               #
# #--------------------------------------------------------#

# dataset = BraTs_Dataset("Brats_2018_data/Brats_2018_data",path_ext = ["/HGG","/LGG"],size=size,apply_transform=False)

# train_split = 0.7
# validation_split = 0.1
# test_split = 0.2

# # percentage amount to split the training set by (in all data there are 200 patients within the training dataset)
# # i.e 0.1 would output 10% of the dataset for a total of 20 patients; whereas 1 would output 100% of the total dataset.
# split_amount = 1

# data_size = len(dataset)
# patients_number = data_size / 155

# train_length = int(155*(np.ceil(patients_number * train_split)))
# validation_length = int(155*(np.floor(patients_number * validation_split)))
# test_length = int(155*(np.floor(patients_number * test_split)))

# # splits the dataset
# split_1 = list(range(0,int(155*(np.ceil((train_length / 155) * split_amount)))))

# train_range = list(range(0,train_length))
# val_range = list(range(train_length,train_length+validation_length))
# #test_range = range(train_length+validation_length,train_length+validation_length+test_length)

# train_data_m = torch.utils.data.RandomSampler(train_range,False)
# validation_data_m = torch.utils.data.RandomSampler(val_range,False)
# #test_data_m = torch.utils.data.SubsetRandomSampler(test_range)

# data_split_m = torch.utils.data.RandomSampler(split_1,False)

# # https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
# print("Training: ", len(train_data_m))
# print("Actual_input: ", len(split_1))
# print("validation: ", len(validation_data_m))

# Train_data=DataLoader(
#     dataset=dataset,
#     batch_size=batch_size,
#     sampler=split_1)

# Val_data=DataLoader(
#     dataset=dataset,
#     batch_size=batch_size,
#     sampler=validation_data_m)

# Train_loss, validation_loss = train(Train_data, Val_data, load=False)

import nibabel as nib

unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
print(unet)

unet_opt = torch.optim.Adam(unet.parameters(), lr=lr,betas=(0.9, 0.999), weight_decay=1e-8)


checkpoint = torch.load("Checkpoints_RANO/Unet_H16_M11_O10/checkpoint_6.pth")

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])
    
img_a = nib.load("Brats_2018_data/Brats_2018_data/HGG/Brats18_CBICA_ATF_1/Brats18_CBICA_ATF_1_whimg_norm.nii.gz")
image = img_a.get_fdata()
print("sHAPE", image.shape)

image = torch.from_numpy(image).unsqueeze(0)

image = image.to(device)
image = image.float() 
image = image.squeeze()
image.requires_grad_()

output = unet(image)

# Catch the output
output_idx = output.argmax()
output_max = output[0, output_idx]

# Do backpropagation to get the derivative of the output based on the image
output_max.backward()

saliency, _ = torch.max(X.grad.data.abs(), dim=1) 
saliency = saliency.reshape(240, 240)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
ax[0].axis('off')
ax[1].imshow(saliency.cpu(), cmap='hot')
ax[1].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()