import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from Unet_modules.RANO_dataloader_2 import BraTs_Dataset
from Unet_modules.dataloader_test import Test_Dataset
#import Net.Unet_Rano_components as net
import Net.ResNet_components as net
import csv
from os import walk
import nibabel as nib
import os
from sklearn.metrics import jaccard_score
from matplotlib.path import Path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def MSELossorthog(output, target):
    
    output_val = output.data.cpu().numpy()
    
    l1 = np.sqrt(np.square(output_val[1]-output_val[3]) + np.square(output_val[0]-output_val[2]))
    l2 = np.sqrt(np.square(output_val[5]-output_val[7]) + np.square(output_val[4]-output_val[6]))
    
    m1 = (abs(output_val[1]/l1-output_val[3]/l1))/(abs(output_val[0]/l1-output_val[2]/l1)+0.1)
    m2 = (abs(output_val[5]/l2-output_val[7]/l2))/(abs(output_val[4]/l2-output_val[6]/l2)+0.1)

    orthog = abs(np.dot(m1,m2))
    
    weight = 2
    
    loss = torch.mean((output - target)**2) + (orthog * weight)
    return loss

#https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
#https://hewjunwei.wordpress.com/2013/01/26/obb-generation-via-principal-component-analysis/
#https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

def Obb(input_array):
    
    input_array = input_array.detach().cpu().numpy()
        
    input_data = np.array([(input_array[1], input_array[0]),
                           (input_array[5], input_array[4]), 
                           (input_array[3], input_array[2]), 
                           (input_array[7], input_array[6])])
    
    input_covariance = np.cov(input_data,y = None, rowvar = 0,bias = 1)
    
    v, vect = np.linalg.eig(input_covariance)
    tvect = np.transpose(vect)
    #use the inverse of the eigenvectors as a rotation matrix and
    #rotate the points so they align with the x and y axes
    rotate = np.dot(input_data,vect)
    
    # get the minimum and maximum x and y 
    mina = np.min(rotate,axis=0)
    maxa = np.max(rotate,axis=0)
    diff = (maxa - mina)*0.5
    
    # the center is just half way between the min and max xy
    center = mina + diff
    
    #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],
                        center+[ diff[0],-diff[1]],
                        center+[ diff[0], diff[1]],
                        center+[-diff[0], diff[1]],
                        center+[-diff[0],-diff[1]]])
    
    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)
    
    return corners, center

def mask(shape,corners):
    nx, ny = shape
    
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    
    points = np.vstack((x,y)).T
    
    path = Path(corners)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))
    
    return grid

c_file = "resnet_data/"

np.set_printoptions(precision=4)

#torch.manual_seed(0)
#np.random.seed(0)

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
#criterion = nn.MSELoss()
criterion = MSELossorthog

n_epochs = 50
input_dim = 4
label_dim = 8
hidden_dim = 32

display_step = 200
batch_size = 16
lr = 0.0001
initial_shape = int(240 * size)
target_shape = int(8)
device = 'cuda'

val_percent = 0.1
test_percent = 0.2
train_percent = 1 - (val_percent + test_percent)

# https://nipy.org/nibabel/gettingstarted.html

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
                
            corners_truth, center_truth = Obb(label_input[input_val,:])
            mask_truth = mask((240,240),corners_truth)*1
            corners_pred, center_pred = Obb(pred[input_val,:])
            mask_pred = mask((240,240),corners_pred)*1

            if np.sum(np.sum(mask_pred)) > 2:
                jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
            else:
                jaccard_val.append(0)
        
        cur_step += 1
    print("Validation complete")
    print(" ")
    
    return losses, jaccard_val


#               Define validation end                    #
#--------------------------------------------------------#
#                Define model start                      #

def train(Train_data,Val_data,load=False):
    
    unet = net.ResNet34().to(device)
    print(unet)
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
                
                corners_truth, center_truth = Obb(label_input[input_val,:])
                mask_truth = mask((240,240),corners_truth)*1
                corners_pred, center_pred = Obb(pred[input_val,:])
                mask_pred = mask((240,240),corners_pred)*1
                
                if np.sum(np.sum(mask_pred)) > 2:
                    jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    jaccard.append(0)
            
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
                
                show_tensor_images(truth_input[:,1,:,:], size=(1, initial_shape, initial_shape),
                                   title="Flair Input Channel ( channel 2 of 4 )")
                plt.show()
                print(label_input[0,:].shape)
                #for i in range(cur_batch_size):
                #    print("input", label_input[i,:].data.cpu().numpy())
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

        
        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints_RANO/" + c_file + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)

        with open("Checkpoints_RANO/" + c_file + "epoch_" + str(epoch) + "training_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        epoch_val_loss, epoch_jaccard_valid = Validate(unet, criterion, Val_data)
        
        valid_loss.append(epoch_val_loss)
        valid_jaccard.append(epoch_jaccard_valid)
        
        with open("Checkpoints_RANO/" + c_file + "epoch_" + str(epoch) + "validation_loss", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
        with open("Checkpoints_RANO/" + c_file + "epoch_" + str(epoch) + "jaccard_index", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(jaccard)
            
        with open("Checkpoints_RANO/" + c_file + "epoch_" + str(epoch) + "validation_jaccard_index", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_jaccard)
            
        #for t_loss_count in range(len(total_loss)):
        t.append(np.mean(total_loss[len(total_loss)-1]))
            
        #for v_loss_count in range(len(valid_loss)):
        v.append(np.mean(valid_loss[len(valid_loss)-1]))

        plt.plot(range(len(t)),t)
        plt.plot(range(len(v)),v)
        plt.legend(["training","validation"])
        plt.show()

    print('Finished Training Dataset')
    return total_loss, valid_loss

#               step and loss output start               #
#--------------------------------------------------------#

dataset = BraTs_Dataset("Brats_2018_data_split/Training",path_ext = ["/HGG","/LGG"],size=size,apply_transform=False)

#dataset = BraTs_Dataset("Brats_2018 data", path_ext = ["/HGG_single_2"],size=size,apply_transform=True)

Validation_dataset = BraTs_Dataset("Brats_2018_data_split/Validation", path_ext=["/HGG","/LGG"],size=size,apply_transform=False)

#Validation_dataset = BraTs_Dataset("Brats_2018 data", path_ext = ["/HGG_single_2"],size=size,apply_transform=True)

#Testing_dataset = BraTs_Dataset("Brats_2018_data_split/Testing", path_ext=["/HGG","/LGG"],size=size,apply_transform=False)

print("Training: ", len(dataset))
print("validation: ", len(Validation_dataset))
#print("Test: ", len(Testing_dataset))

Train_data = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True)

Val_data = DataLoader(
    dataset=Validation_dataset,
    batch_size=batch_size,
    shuffle=True)

#Test_data = DataLoader(
#    dataset=Testing_dataset,
#    batch_size=batch_size,
#    shuffle=False,
#    drop_last=True)

Train_loss, validation_loss = train(Train_data, Val_data, load=False)

# i need to record the outputs alongside the model parameters in an addtional file so that i can delete the data outputs that are taking up space
#save progress to github - its about time you did that tbh
#maybe split some of the models into pre-packaged data model/files so that i dont lose progress here