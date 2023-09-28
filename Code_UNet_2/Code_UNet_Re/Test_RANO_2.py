import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from Unet_modules.RANO_dataloader_2_scandir import BraTs_Dataset
from Unet_modules.dataloader_test import Test_Dataset
from Unet_modules.Penalty import Penalty
from Unet_modules.Evaluation import Jaccard_Evaluation as Jacc

import Net.Unet_Rano_components as net
from os import walk
import nibabel as nib
import os

import seaborn as sns
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.metrics import jaccard_score

sns.set_theme()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
criterion = Penalty.MSELossorthog

n_epochs = 6
input_dim = 4
label_dim = 8
hidden_dim = 16

display_step = True
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(8)
device = 'cuda'
outname = "Unet_H16_M12_O10"

def Test(Test_data, unet, unet_opt, path, path_ext,output_plot = True):
    
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
    savevalue = 0
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
                
                corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
                mask_truth = Jacc.mask((initial_shape,initial_shape),corners_truth)*1
                corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
                mask_pred = Jacc.mask((initial_shape,initial_shape),corners_pred)*1
                
                if np.sum(np.sum(mask_pred)) > 2:
                    jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    jaccard.append(float("NaN"))
            
            if display_step == True:
                #print(jaccard[-16:])  
                for i in range(cur_batch_size):
                    print("______________________________________")
                    print("Jaccard score:", jaccard[-(16-i)])  
                    
                    #print("prediction",pred[i,:].data.cpu().numpy())
                    plt.grid(False)
                    plt.imshow(truth_input[i,1,:,:].data.cpu().numpy(),cmap='gray')

                    data_in = label_input[i,:].data.cpu().numpy()
                    D3 = np.asarray([[data_in[1],data_in[3]],[data_in[0],data_in[2]]]) 
                    D4 = np.asarray([[data_in[5],data_in[7]],[data_in[4],data_in[6]]]) 

                    plt.plot(D3[0, :], D3[1, :], lw=3, c='y',label='_nolegend_')
                    plt.plot(D4[0, :], D4[1, :], lw=3, c='y',label='Ground Truth')

                    data_out = pred[i,:].data.cpu().numpy()
                    D1 = np.asarray([[data_out[1],data_out[3]],[data_out[0],data_out[2]]]) 
                    D2 = np.asarray([[data_out[5],data_out[7]],[data_out[4],data_out[6]]]) 

                    plt.plot(D1[0, :], D1[1, :], lw=2, c='b',label='_nolegend_')
                    plt.plot(D2[0, :], D2[1, :], lw=2, c='b',label='Prediction')

                    plt.legend(loc='best')
                    plt.title("Jaccard score of " + str('%.2f' % jaccard[-(16-i)]))
                                        
                    # here we need to mae the d[data_val] a folder in the file to hold the images this would make it easier, maybe even zip it up? to save on space.
                    plt.savefig('Comp_images_2/' + outname +'/' + d[data_val] + '_' +'_image_' +  str(savevalue) + "_" + str(jaccard[-(16-i)]) +'.png')
                    savevalue = savevalue + 1
                    plt.show()
                    plt.clf()
                    plt.cla()

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
                    print(ext)
                    print(d)
                    print(data_val)
                    np.savez("Predictions_RANO/newtest/" + d[data_val] + "_" + ext,RANO=pred_out)

                    data_val += 1
                    pred_out = []
                    img_num = 0
    return jaccard
                    
#dataset = Test_Dataset("MICCAI_BraTS_2018_Data_Validation",path_ext=["/data"],size=size,apply_transform=False)
dataset = BraTs_Dataset("Brats_2018_data/Brats_2018_data",path_ext=["/HGG","/LGG"],size=size,apply_transform=False)

Data_1 = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False)

unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

checkpoint = torch.load("Checkpoints_RANO/" + outname + "/checkpoint_34.pth")

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

jaccard = Test(Data_1, unet, unet_opt,"Brats_2018_data/Brats_2018_data",["HGG","LGG"])

a = []
a.extend(input_data())

figure(figsize=(8, 6), dpi=80)
figure(figsize=(10,10))

print(np.array(jaccard).shape,np.array(a).shape)
plt.scatter(np.array(jaccard),np.array(a))
plt.xlabel("tumour size (pixels)")
plt.ylabel("Jaccard index")
plt.show()

# Data_1 = pd.DataFrame(data=jaccard, columns=range(1)).assign(Data="data")
# mdf = pd.melt(Data_1, id_vars=['Data'])

# ax = sns.boxplot(x="Data", y="value", data=mdf)  # RUN PLOT   
# plt.show()
# ax = sns.violinplot(x="Data", y="value", data=mdf)  # RUN PLOT   
# plt.show()