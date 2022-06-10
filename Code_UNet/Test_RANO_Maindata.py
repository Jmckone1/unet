from Unet_modules.RANO_dataloader_2_scandir_Test import BraTs_Dataset
from Unet_modules.Evaluation import Jaccard_Evaluation as Jacc

# this was used for the official validation dataset - will need to check and confirm this
#from Unet_modules.dataloader_test import Test_Dataset
# from Unet_modules.Penalty_3 import Penalty
#from Unet_modules.Penalty import Penalty
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import walk

import Net.Unet_Rano_components as net
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import nibabel as nib
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os
import Unet_modules.Parameters as Param

# this code uses a fixed 155 value for each of the images which might need fixing at some point, for the time being it suits the purpose with being ran on the full non reduced dataset.

sns.set_theme()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= Param.Global.GPU

# these lines commented can be removed after testing if it runs without them.

# size = Param.rNet.size
# input_dim = Param.rNet.input_dim
# label_dim = Param.rNet.label_dim
# hidden_dim = Param.rNet.hidden_dim
# lr = Param.rNet.lr
# batch_size = Param.rNet.batch_size
# initial_shape = Param.rNet.initial_shape
# target_shape = Param.rNet.target_shape
# device = Param.rNet.device
# load_path = Param.test_rNet.dataset_path # "Brats_2018_data/Brats_2018_data"
# load_path_ext = Param.rNet.Extensions # ["/HGG","/LGG"]
# display_step = Param.test_rNet.display_step
# outname = Param.test_rNet.output_path
# checkpoint_path = Param.test_rNet.checkpoint_path
# Rano_save_path = Param.test_rNet.Rano_save_path
# image_save_path = Param.test_rNet.image_save_path
# if not os.path.exists(Param.test_rNet.Rano_save_path):
#     os.makedirs(Param.test_rNet.Rano_save_path)
#     os.makedirs(image_save_path)
# criterion, mse, cosine = Penalty.MSELossorthog

##############################################################
##############################################################
##############################################################

def input_value_count(path,path_ext):
    d = []
    index_max = []
    index_max.extend([0])

    path_ext = path_ext
    HGG_len = 0
    c_s = 0

    for input_ in range(len(path_ext)):
        counter = 0
        # each folder in extension
        for files in os.scandir(path + path_ext[input_]):
            if files.is_dir() or files.is_file():
                if not files.name.startswith("."):
                    d.append(path_ext[input_] + "/" + files.name)
        counter = len(d)
        # if the index file does not exist then create a new one, else load the existing one.
        # may have to implement an override in the case of a necessary deletion.
        if not os.path.exists(path + Param.rData_Test.index_file):
            print("Creating index_file...")
            for directory in tqdm(range(counter-c_s)):
                if directory == 0:
                    if input_ == 0:
                        c_s = counter
                if input_ == 1:
                    directory = directory + c_s

                file = d[directory] + '/' + d[directory] + "_" + Param.rData_Test.image_in + '.nii.gz'
                full_path = os.path.join(path + path_ext[input_], file)
                img_a = nib.load(full_path)
                img_data = img_a.get_fdata()

                index_max.extend([img_data.shape[3] + index_max[-1]])

            if input_ == len(path_ext):
                print("Saving index file . . . ")
                np.save(path + Param.rData_Test.index_file, index_max)
                print("Index file complete")
        else:
            index_max = np.load(path + Param.rData_Test.index_file)

            # value for extension swapping
            if input_ == 0:
                HGG_len = index_max[-1]  
    
    return d, HGG_len

def test_main(Train_data, checkpoint_path, load_path, load_path_ext, display_step=True):
    
    d, HGG_len = input_value_count(load_path, load_path_ext)
    # print(d)
    
    unet = net.UNet(Param.rNet.input_dim, Param.rNet.label_dim, Param.rNet.hidden_dim).to(Param.rNet.device)
    
    #lr, weight decay, betas need to be inclued in the param file
    unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.rNet.lr, betas=Param.rNet.Betas, weight_decay=Param.rNet.Weight_Decay)
    
    # load model parameters for the testing
    checkpoint = torch.load(checkpoint_path)
    unet.load_state_dict(checkpoint['state_dict'])
    unet_opt.load_state_dict(checkpoint['optimizer'])
    unet.eval()
    
    pred_out = []
    jaccard = []
    data_val = 0
    img_num = 0

    for truth_input, label_input in tqdm(Train_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        
        # print(truth_input.shape)
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

        for input_val in range(cur_batch_size):

            corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
            mask_truth = Jacc.mask((240,240),corners_truth)*1
            corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
            mask_pred = Jacc.mask((240,240),corners_pred)*1

            if np.sum(np.sum(mask_pred)) > 2:
                jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
            else:
                jaccard.append(float("NaN"))
                
        pred_output = pred.cpu().detach().numpy()
        
        if display_step == True:
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
                
                pred_out = np.append(pred_out, pred_output[i,:])
                
                if img_num == 155:
                    
                    # assign the correct extension - HGG or LGG
                    if data_val < HGG_len:
                        ext = load_path_ext[0]
                    else:
                        ext = load_path_ext[1]
                        
                    print("RANO", Param.test_rNet.Rano_save_path  + "/" + d[data_val] + "/")
                    if not os.path.exists(Param.test_rNet.Rano_save_path  + "/" + d[data_val] + "/"):
                        os.makedirs(Param.test_rNet.Rano_save_path  + "/" + d[data_val] + "/")
                        
                    np.savez(Param.test_rNet.Rano_save_path  + "/" + d[data_val] + "/", RANO=pred_out)
                    
                    print("Saving " + str(d[data_val]) + " RANO . . . ")
                    data_val += 1
                    pred_out = []
                    img_num = 0
                    
                    if not os.path.exists(Param.test_rNet.image_save_path + "/" + d[data_val] + "/"):
                        os.makedirs(Param.test_rNet.image_save_path + "/" + d[data_val] + "/")
                    plt.savefig(Param.test_rNet.image_save_path + "/" + d[data_val] + "/" +'Slice_' +  str(img_num) + "_" + str(jaccard[-(16-i)]) +'.png')
                    
                    plt.show()
                    plt.clf()
                    plt.cla()
                
                else:
                    
                    # here we need to make the d[data_val] a folder in the file to hold the images this would make it easier, maybe even zip it up? to save on space.
                    # if filename doesnt exist then make new one for /d[data_val]
                    if not os.path.exists(Param.test_rNet.image_save_path + "/" + d[data_val] + "/"):
                        os.makedirs(Param.test_rNet.image_save_path + "/" + d[data_val] + "/")

                    plt.savefig(Param.test_rNet.image_save_path + "/" + d[data_val] + "/" +'Slice_' +  
                                str(img_num) + "_" + str(jaccard[-(16-i)]) +'.png')
                    plt.show()
                    plt.clf()
                    plt.cla()
                    img_num += 1
            

    print('Finished Training Dataset')
    return jaccard

#               step and loss output start               #
#--------------------------------------------------------#

dataset = BraTs_Dataset(Param.test_rNet.dataset_path, path_ext = Param.rNet.Extensions, size=Param.rNet.size, apply_transform=False)
print(Param.test_rNet.output_path)
##################################################################################################################################
# dataset length splitting ######################################################################## ##############################
##################################################################################################################################
index_f = np.load(Param.test_rNet.dataset_path + Param.rData_Test.index_file)
patients_number = len(index_f)

train_length = index_f[int(np.floor(patients_number*Param.rNet.train_split))]
validation_length = index_f[int(np.ceil(patients_number*Param.rNet.validation_split))]
test_length = index_f[int(np.ceil(patients_number*Param.rNet.test_split))-1]
# all_data_length = index_f[-1]
# custom_split = index_f[int(np.ceil(patients_number*Param.rNet.custom_split_amount))-1]

# train_range = list(range(0,train_length))
val_range = list(range(train_length,train_length+validation_length))
# test_range = list(range(train_length+validation_length,train_length+validation_length+test_length -1))
# all_data_range = list(range(0,all_data_length))
# custom_split_range = list(range(0,custom_split))

# train_data_m = torch.utils.data.RandomSampler(train_range,False)
# validation_data_m = torch.utils.data.RandomSampler(val_range,False)
# test_data_m = torch.utils.data.SubsetRandomSampler(test_range)
# all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
# custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)
##################################################################################################################################

Test_data=DataLoader(
    dataset=dataset,
    batch_size=Param.rNet.batch_size,
    sampler=val_range)

test_jaccard = test_main(Test_data, Param.test_rNet.checkpoint_path, Param.test_rNet.dataset_path, Param.rNet.Extensions, Param.test_rNet.display_step)
