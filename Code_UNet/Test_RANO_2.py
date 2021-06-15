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
import Net.Unet_Rano_components as net
import csv
from os import walk
import nibabel as nib
import os

import seaborn as sns
sns.set_theme()
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.metrics import jaccard_score
from matplotlib.path import Path

# put the additional functions in an external file and call them !!!
def input_data():
    path_ext = ["/HGG","/LGG"]
    path ="Brats_2018_data_split/Validation"
    d = []
    index_max = []
    index_max.extend([0])
    f = []

    path_ext = path_ext

    c_s = 0

    # each extension - HGG or LGG
    for input_ in range(len(path_ext)):
        counter = 0
        # each folder in extension
        for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
            # gets rid of any pesky leftover .ipynb_checkpoints files
            if not dir_names == []:
                if not dir_names[0].startswith("."):

                    d.extend(dir_names)

                    counter = len(d)
                    #print("D:",len(d))

        for directory in range(counter-c_s):
            if directory == 0:
                if input_ == 0:
                    c_s = counter
            if input_ == 1:
                directory = directory + c_s

            file = d[directory] + '/' + d[directory] + "r_" + "whseg" + '.nii.gz'
            full_path = os.path.join(path + path_ext[input_], file)
            img_a = nib.load(full_path)
            img_data = img_a.get_fdata()

            #print(img_data.shape)
            index_max.extend([img_data.shape[2] + index_max[-1]])
            for xc in range(img_data.shape[2]):
                f.extend([np.sum(np.sum(img_data[:,:,xc]))])

            #print(index_max)
            # value for extension swapping
            if input_ == 0:
                HGG_len = index_max[-1]
        #print(HGG_len)
        #print(index_max)


    # inputs to global
    count = index_max[-1] # anything with 155 in it needs to be redone to not rely on the hard coded value

    return f

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

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
criterion = MSELossorthog

n_epochs = 6
input_dim = 4
label_dim = 8
hidden_dim = 32

display_step = True
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(8)
device = 'cuda'

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

def Test(Test_data, unet, unet_opt, path, path_ext):
    
    f = []
    d = []
    HGG_len = 0
    
    counter = 0
    # each folder in extension
    for i in range(2):
        for (dir_path, dir_names, file_names) in walk(path + "/" + path_ext[i]):

            # gets rid of any pesky leftover .ipynb_checkpoints files
            if not dir_names == []:
                if not dir_names[0].startswith("."):

                    f.extend(file_names)
                    d.extend(dir_names)
                    counter = len(d)

            HGG_len = counter * 155

    print(path + path_ext[0])

    unet.eval()
    
    img_num = 0 # file size output
    pred_out = [] # prediction array output
    data_val = 0 # number for file output naming
    jaccard = []
    
    for truth_input,label_input in tqdm(Test_data):

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
            
            for input_val in range(cur_batch_size):
                
                corners_truth, center_truth = Obb(label_input[input_val,:])
                mask_truth = mask((240,240),corners_truth)*1
                corners_pred, center_pred = Obb(pred[input_val,:])
                mask_pred = mask((240,240),corners_pred)*1
                
                if np.sum(np.sum(mask_pred)) > 2:
                    jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    jaccard.append(float("NaN"))
            
            if display_step == True:
                print(jaccard[-16:])  
                for i in range(cur_batch_size):
                    print("prediction",pred[i,:].data.cpu().numpy())
                    
                    plt.imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')
                    
                    data_in = label_input[i,:].data.cpu().numpy()
                    D3 = np.asarray([[data_in[1],data_in[3]],[data_in[0],data_in[2]]]) 
                    D4 = np.asarray([[data_in[5],data_in[7]],[data_in[4],data_in[6]]]) 
                    
                    plt.plot(D3[0, :], D3[1, :], lw=3, c='y')
                    plt.plot(D4[0, :], D4[1, :], lw=3, c='y')
                    
                    data_out = pred[i,:].data.cpu().numpy()
                    D1 = np.asarray([[data_out[1],data_out[3]],[data_out[0],data_out[2]]]) 
                    D2 = np.asarray([[data_out[5],data_out[7]],[data_out[4],data_out[6]]]) 
                    
                    plt.plot(D1[0, :], D1[1, :], lw=2, c='b')
                    plt.plot(D2[0, :], D2[1, :], lw=2, c='b')

                    plt.show()

            pred_val = pred.cpu().detach().numpy()
            #print(pred_val.shape)
            
            for i in range(cur_batch_size):
                
                pred_out = np.append(pred_out, pred_val[i,:])

                img_num += 1

                if img_num == 155:
                    
                    # assign the correct extension - HGG or LGG
                    if data_val < HGG_len:
                        ext = path_ext[0]
                    else:
                        ext = path_ext[1]
                    #print(ext)
                    #print(d)
                    #print(data_val)
                    #np.savez("Predictions_RANO/" + d[data_val] + "_" + ext,RANO=pred_out)

                    data_val += 1
                    pred_out = []
                    img_num = 0
    return jaccard
                    
#dataset = Test_Dataset("MICCAI_BraTS_2018_Data_Validation",path_ext=["/data"],size=size,apply_transform=False)
                             
dataset = BraTs_Dataset("Brats_2018_data_split/Validation",path_ext=["/LGG","/HGG"],size=size,apply_transform=False)

Data_1 = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False)

unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

checkpoint = torch.load("Checkpoints_RANO/unet8_data/checkpoint_49.pth")

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

jaccard = Test(Data_1, unet, unet_opt,"Brats_2018_data_split/Validation",["HGG","LGG"])

a = []
a.extend(input_data())
#print(a)

figure(figsize=(8, 6), dpi=80)
figure(figsize=(10,10))

#this needs to be the jaccard for the validation in order for the index values to be correct, or i need to track it throughout training (which is probably more effort than it is worth...)
print(np.array(jaccard).shape,np.array(a).shape)
plt.scatter(np.array(jaccard),np.array(a))
plt.xlabel("tumour size (pixels)")
plt.ylabel("Jaccard index")
plt.show()

Data_1 = pd.DataFrame(data=jaccard, columns=range(1)).assign(Data="data")
mdf = pd.melt(Data_1, id_vars=['Data'])

ax = sns.boxplot(x="Data", y="value", data=mdf)  # RUN PLOT   
plt.show()
ax = sns.violinplot(x="Data", y="value", data=mdf)  # RUN PLOT   
plt.show()