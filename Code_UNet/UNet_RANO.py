import torch
# from torch import nn # need this to use built in loss functions
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from Unet_modules.RANO_dataloader_2 import BraTs_Dataset
from Unet_modules.dataloader_test import Test_Dataset
from Unet_modules.Penalty import Penalty
from Unet_modules.Evaluation import Jaccard_Evaluation as Jacc

import Net.Unet_Rano_components as net

import csv
import os
from sklearn.metrics import jaccard_score

np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# In the format "FileName/"
c_file = "Unet_H16_M9_O4A0_V2/"

np.set_printoptions(precision=4)

# image interpolation multiplier
size = 1

n_epochs = 100
input_dim = 4
label_dim = 8
hidden_dim = 16
orth_penalty = 4
area_penalty = 0

Improvement = 0

#criterion = nn.MSELoss()
loss_f = Penalty(orth_penalty,area_penalty)
criterion = loss_f.MSELossorthog

display_step = 200
batch_size = 16
lr = 0.0001
initial_shape = int(240 * size)
target_shape = int(8)
device = 'cuda'

val_percent = 0.1
test_percent = 0.2
train_percent = 1 - (val_percent + test_percent)

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

def train(Train_data,Val_data,load=False):
    
    unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
    print(unet)
    
    if not os.path.exists("Checkpoints_RANO/" + c_file):
        os.makedirs("Checkpoints_RANO/" + c_file)
    
    with open("Checkpoints_RANO/" + c_file + "Model_architecture", 'w') as write: 
        #write = csv.writer(f)
        write.write("epochs: " + str(n_epochs) + "\n")
        write.write("batch size: " + str(batch_size) + "\n")
        write.write("learning rate: " + str(lr) + "\n")
        write.write("orthogonality weight: " + str(orth_penalty) + "\n")
        write.write("area weight: " + str(area_penalty) + "\n")

        write.write(str(unet))
    
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr,betas=(0.9, 0.999), weight_decay=1e-8)

    if load == True:
        checkpoint = torch.load("Checkpoints_RANO/" + c_file + "checkpoint_0_step_1900.pth")

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #
    t = []
    v = []

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
        jaccard = []
        valid_jaccard = []
        
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
            
           # print(pred.shape)
           # input("")

            # forward
            unet_loss = criterion(pred, label_input)
            
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
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item() * truth_input.size(0)
            
            cur_step += 1

#                     Run model end                      #
#--------------------------------------------------------#         
#                  Display stage start                   #

            loss_values.append(running_loss / len(Train_data))
            total_loss.append(loss_values)
        
            if cur_step % display_step == 0:

                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                
#                 show_tensor_images(truth_input[:,1,:,:], size=(1, initial_shape, initial_shape),
#                                    title="Flair Input Channel ( channel 2 of 4 )")
                plt.show()
                print(label_input[0,:].shape)
                #for i in range(cur_batch_size):
                #    print("input", label_input[i,:].data.cpu().numpy())
                
                # Print jaccard for current output in the batch
                print("index", jaccard[-cur_batch_size:]) 
                print("")
                    
                for i in range(cur_batch_size):
                    
                    print("input", label_input[i,:].data.cpu().numpy())
                    print("prediction",pred[i,:].data.cpu().numpy())
                    
                    f, axarr = plt.subplots(1,2)

                    data_in = label_input[i,:].data.cpu().numpy()
                    D1 = np.asarray([[data_in[1],data_in[3]],[data_in[0],data_in[2]]]) 
                    D2 = np.asarray([[data_in[5],data_in[7]],[data_in[4],data_in[6]]]) 
                    
                    axarr[0].imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')
                    axarr[0].plot(D1[0, :], D1[1, :], lw=2, c='r')
                    axarr[0].plot(D2[0, :], D2[1, :], lw=2, c='b')
                    axarr[0].set_title('Truth')
                    
                    data_out = pred[i,:].data.cpu().numpy()
                    D1 = np.asarray([[data_out[1],data_out[3]],[data_out[0],data_out[2]]]) 
                    D2 = np.asarray([[data_out[5],data_out[7]],[data_out[4],data_out[6]]]) 

                    axarr[1].imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')
                    axarr[1].plot(D1[0, :], D1[1, :], lw=2, c='r')
                    axarr[1].plot(D2[0, :], D2[1, :], lw=2, c='b')
                    axarr[1].set_title('Prediction')
                    
                    plt.show()
                   
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
        epoch_val_loss, epoch_jaccard_valid = Validate(unet, criterion, Val_data)
        
        valid_loss.append(epoch_val_loss)
        valid_jaccard.append(epoch_jaccard_valid)
        
        if np.nanmean(epoch_jaccard_valid) > Improvement:
            Improvement = np.nanmean(epoch_jaccard_valid)
        
            print("saving epoch: ", epoch)
            checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
            out = "Checkpoints_RANO/" + c_file + "checkpoint_" + str(epoch) + ".pth"
            torch.save(checkpoint, out)

        with open("Checkpoints_RANO/" + c_file + "Training_loss/epoch_" + str(epoch) + "training_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)

        with open("Checkpoints_RANO/" + c_file + "Validation_loss/epoch_" + str(epoch) + "validation_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)

        with open("Checkpoints_RANO/" + c_file + "Training_Jaccard/epoch_" + str(epoch) + "jaccard_index", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(jaccard)

        with open("Checkpoints_RANO/" + c_file + "Valdiation_Jaccard/epoch_" + str(epoch) + "validation_jaccard_index", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_jaccard)

#             for t_loss_count in range(len(total_loss)):
#             t.append(np.mean(total_loss[len(total_loss)-1]))

#             #for v_loss_count in range(len(valid_loss)):
#             v.append(np.mean(valid_loss[len(valid_loss)-1]))

#             plt.plot(range(len(t)),t)
#             plt.plot(range(len(v)),v)
#             plt.legend(["training","validation"])
#             plt.show()

    print('Finished Training Dataset')
    return total_loss, valid_loss

#               step and loss output start               #
#--------------------------------------------------------#

Train_dataset = BraTs_Dataset("Brats_2018_data_split/Training", path_ext = ["/HGG","/LGG"],size=size,apply_transform=False)

Validation_dataset = BraTs_Dataset("Brats_2018_data_split/Validation", path_ext=["/HGG","/LGG"],size=size,apply_transform=False)

print("Training: ", len(Train_dataset))
print("validation: ", len(Validation_dataset))

Train_data = DataLoader(
    dataset=Train_dataset,
    batch_size=batch_size,
    shuffle=True)

Val_data = DataLoader(
    dataset=Validation_dataset,
    batch_size=batch_size,
    shuffle=True)

Train_loss, validation_loss = train(Train_data, Val_data, load=False)