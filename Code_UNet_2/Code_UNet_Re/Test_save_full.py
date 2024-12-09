# from Net_modules.Unet_Main_dataloader import BraTs_Dataset
from Net_modules.Loading_Test_data import Load_Dataset
import Net_modules.Model_hyperparameters as Param

from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import DiceLoss
import Net.Unet_components_split as net

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
from torch import nn
import numpy as np
import shutil
import torch
import csv
import os

import sys
import random
import seg_metrics.seg_metrics as sg
import time
                
from itertools import chain
from collections import defaultdict
                    

print("Loading seed . . .")

seed = Param.Parameters.Network["Global"]["Seed"]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(Param.Parameters.Network["Global"]["GPU"])
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

torch.cuda.empty_cache()
os.chdir(os.getcwd())

headers = ["Global","Hyperparameters","Train_paths"]
print("Torch version: ",torch.__version__)
print("")
for h in headers:
    for key, value in Param.Parameters.Network[h].items():
        print(f'{key: <30}{str(value): <35}')

criterion = nn.BCEWithLogitsLoss()

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# defining a dictionary class to save the output files.
class Define_dictionary(dict):
    def __init__(self):
        self = dict()
    def add(self, key, value):
        self[key] = value
        
# Function to define the dice score of a volume - taken and adapted from the deepmind project code by recommendation of project supervisor.
def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice

# Test save main function for testing the volumes for output and then saving all the values based on the param file.

def Test_save(Test_data, mask_names, unet, unet_opt, path, path_ext, save_path, load_path, volume_shape_list, save=False, save_image = False, save_val =""):

    current_volume = 0
    
    original = Param.Parameters.Network["Global"]["Param_location"]
    target = "Test_outputs/" + Param.Parameters.Network["Test_paths"]["Save_path"] + "Model_hyperparameters.py"
    if not os.path.exists("Test_outputs/" + Param.Parameters.Network["Test_paths"]["Save_path"]):
        os.makedirs("Test_outputs/" + Param.Parameters.Network["Test_paths"]["Save_path"])
        
    shutil.copyfile(original, target)
    
    Dice_output = Define_dictionary()
    Dice_output_sigmoid = Define_dictionary()
    
    volume_predi = np.empty((Param.Parameters.Network["Hyperparameters"]["Image_size"][0],
                             Param.Parameters.Network["Hyperparameters"]["Image_size"][1],
                             volume_shape_list[current_volume]))
    volume_truth = np.empty((Param.Parameters.Network["Hyperparameters"]["Image_size"][0],
                             Param.Parameters.Network["Hyperparameters"]["Image_size"][1],
                             volume_shape_list[current_volume]))
    
    # number for file output naming (data_val) and file size output (img_num)
    data_val = 0 
    img_num = 0 
    
    # create the save path if it doesnt exist
    if not os.path.exists(save_path) and save == True:
        os.makedirs(save_path)
        
    # start test 
    unet.eval()
    for truth_input, label_input, dataloader_path in tqdm(Test_data):
        
        cur_batch_size = len(truth_input)
        truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
        truth_input = truth_input.float() 
        
        label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
        label_input = label_input.float()

        if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
            truth_input = truth_input[np.newaxis,:,:,:]
            label_input = label_input[np.newaxis,:,:]
        if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 4:
            truth_input = truth_input.squeeze()
            label_input = label_input.squeeze()
            if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
                truth_input = truth_input[np.newaxis,:,:,:]
                
                label_input = label_input[np.newaxis,:,:]
#             print(np.shape(label_input))
#             input("")
#             print(np.shape(label_input))
            label_input = label_input[:,np.newaxis,:,:]
        
        unet_opt.zero_grad()
        pred = unet(truth_input)
        pred = pred.squeeze()

        if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:
            pred = pred[:,np.newaxis,:,:]

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        if save == True:
            for i in range(cur_batch_size):
                prediction_sigmoid = (pred_output[i,:,:] > 0.5).astype(int)
                
                # convert from batch format to volume format
                volume_predi[:,:,img_num] = prediction_sigmoid[:,:]
                
                mask_path = os.getcwd() + Param.Parameters.Network["Test_paths"]["Data_path"] + "/labelsTr/" + mask_names[data_val] + ".nii.gz"
                mask = nib.load(mask_path).get_fdata()
                volume_truth[:,:,img_num] = mask
                
                
                img_num += 1
                data_val += 1 
#                 print(img_num, "/", volume_shape_list[current_volume])
#                 if img_num == volume_shape_list[current_volume] -1:
#                     input("1 before")
                    
#                 print(img_num, "/", volume_shape_list[current_volume])
                # but a wait in here and maybe below the next line too.
                if img_num == volume_shape_list[current_volume]:
                    
#                     input("end")
#                     print("pred shape", np.shape(volume_predi))
#                     print("Truth shape", np.shape(volume_truth))
                    
#                     print("Vol sum pred", np.sum(np.sum(volume_predi)))
#                     print("Vol sum truth", np.sum(np.sum(volume_truth)))
# #                     input("")
                    # volume_dice_output = calculate_dice((volume_predi > 0.5).astype(int),volume_truth)
    
                    volume_dice_output = Dice_Eval.dice_score((volume_predi > 0.5).astype(int),volume_truth)
        
#                     print(np.min(volume_predi),np.max(volume_predi))
#                     input("checking truth min max")
                    
#                     print(np.min(volume_truth),np.max(volume_truth))
#                     input("checking truth min max")

#                     csv_file = 'metrics_' + str(current_volume) + '.csv'
                    
                    if not os.path.exists(save_path + "/Metrics/") and save == True:
                        os.makedirs(save_path + "/Metrics/")
                   
                    metrics = sg.write_metrics(labels=[1],
                                               gdth_img=volume_truth,
                                               pred_img=(volume_predi > 0.5).astype(int),
                                               csv_file = save_path + "/Metrics/" + dataloader_path[i][46:-7] +
                                                                       "_Metrics.csv", TPTNFPFN=True)
#                     print(metrics)
                    
# #                     print(os.path.join(save_path + dataloader_path[i][46:]))
# #                     print(os.path.join(save_path + dataloader_path[i][46:-7], str(current_volume) + ".csv"))
                    
# #                     input("")
                    
#                     print("Dice_outputs", volume_dice_output)
                
                    # Add the volume path and the volume dice score to the dictionary
                    Dice_output.add(dataloader_path[i], volume_dice_output)
                    
                    # save image prediction
                    if save == True:
                        pred_img_save = nib.Nifti1Image(volume_predi, np.eye(4))
                        if not os.path.exists(os.path.join(save_path)):
                            os.makedirs(os.path.join(save_path))
                        
                        nib.save(pred_img_save, os.path.join(save_path + dataloader_path[i][46:])) 
                                         
            
                    current_volume += 1
#                     print(current_volume, "/", len(volume_shape_list))
                    if current_volume < len(volume_shape_list):
                        volume_predi = np.empty((Param.Parameters.Network["Hyperparameters"]["Image_size"][0],
                                 Param.Parameters.Network["Hyperparameters"]["Image_size"][1],
                                 volume_shape_list[current_volume]))
                        volume_truth = np.empty((Param.Parameters.Network["Hyperparameters"]["Image_size"][0],
                                 Param.Parameters.Network["Hyperparameters"]["Image_size"][1],
                                 volume_shape_list[current_volume]))
                    
                    img_num = 0
            
            # saving the volume name alongside the volume output dice scores for the predictions
            with open(os.path.join(save_path + "_Dice_predictions.csv"), 'w', encoding='UTF8') as f:
                for key, val in Dice_output.items():
                    writer = csv.writer(f)
                    writer.writerow([key, val])
    
################################################################################################################

print("Loading Dataset")
folder = np.loadtxt(os.getcwd() + Param.Parameters.Network["Test_paths"]["Data_path"] + "/Training_dataset.csv", delimiter=",",dtype=str)

image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

dataset = Load_Dataset(os.getcwd() + Param.Parameters.Network["Test_paths"]["Data_path"],
                       image_folder_in,
                       masks_folder_in,
                       Param.Parameters.Network["Hyperparameters"]["Apply_Augmentation"])

volume_shape_list = np.loadtxt(os.getcwd() + "/Datasets/volume_counter_brats_0.csv", delimiter=",",dtype=str)
# volume_shape_list = np.loadtxt(os.getcwd() + "/Datasets/volume_counter_ct_0.csv", delimiter=",",dtype=str)
volume_shape_list = [int(i) for i in volume_shape_list]

print(len(volume_shape_list))
# 
("")

test_volume_list = volume_shape_list[-int(np.floor((len(volume_shape_list) / 10)*2)):]
test_main_size = np.sum(test_volume_list)

# print(test_volume_list)

# print(test_main_size)
# # input("")

Dataset_size = len(folder)
testing = folder[-test_main_size:,2]
testing_split = np.squeeze(testing).astype(int)

print("Total number: ", Dataset_size)
print("Non empty masks: ",len(folder[np.where(folder[:,-1].astype(float) > 0)]))
print("Empty masks: ",len(folder[np.where(folder[:,-1].astype(float) == 0)]))

mask_names = folder[-test_main_size:,1]

del folder

Test_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    num_workers=0,
    sampler=testing_split, 
    pin_memory=True,
    worker_init_fn=_init_fn)

# if Param.Parameters.Network["Hyperparameters"]["Use_weights"] == True:
#     unet = net.UNet.load_weights(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
#                                  Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
#                                  Param.Parameters.Network["Hyperparameters"]["Hidden_dim"], 
#                                  Param.Parameters.Network["Hyperparameters"]["Regress"],
#                                  Param.Parameters.Network["Hyperparameters"]["Allow_update"],
#                                  Param.Parameters.Network["Train_paths"]["Checkpoint_load"]
#                                  ).to(Param.Parameters.Network["Global"]["device"])
# else:
unet = net.UNet(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
                             Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                                 Param.Parameters.Network["Hyperparameters"]["Hidden_dim"]).to(Param.Parameters.Network["Global"]["device"])

unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.Parameters.Network["Hyperparameters"]["Learning_rate"], weight_decay=Param.Parameters.Network["Hyperparameters"]["Weight_decay"])

Save = True

# save for intermediate checkpoints defined between 50 and 550 at each 50 batches
if Param.Parameters.Network["Test_paths"]["Intermediate_checkpoints"] == True:
    values = np.linspace(50, 550, num=11)
    for j in range(len(values)):
        
        print(" ")
        print("Starting prediction of qualitative outputs for batch ", values[j])
        
        load_path = Param.Parameters.Network["Test_paths"]["Load_path"] + "/checkpoint_0" + "_" + str(int(values[j])) + ".pth"
        save_path = Param.Parameters.Network["Test_paths"]["Save_path"] + "/step_0_" + str(int(values[j]))

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data,
                  mask_names,
                  unet, 
                  unet_opt, 
                  Param.Parameters.Network["Test_paths"]["Data_path"], 
                  Param.Parameters.Network["Test_paths"]["Extensions"], 
                  save_path, 
                  load_path, 
                  test_volume_list,
                  save=Save)

# save for at the checkpoints at each end of epoch (in this case 1, 2 and 3)
if Param.Parameters.Network["Test_paths"]["End_epoch_checkpoints"] == True:
    for i in range(Param.Parameters.Network["Test_paths"]["Epochs"]):
        
        print(" ")
        print("Starting prediction of qualitative outputs for end of Epoch ", i + 1)

        load_path = Param.Parameters.Network["Test_paths"]["Load_path"] + "/checkpoint_" + str(i) + ".pth"
        save_path = Param.Parameters.Network["Test_paths"]["Save_path"] + "/Epoch_" + str(i) + "/"
        
        if os.path.exists(save_path):
            print("Path already exists")
            print("Please enter Y to delete file contents or anything else to exit: ")
            replace = input("")
            if replace == "Y":
                shutil.rmtree(save_path)
                print("File deleted . . . continuing script")
            else:
                print("Exiting script")
                sys.exit()

        checkpoint = torch.load(load_path)

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

        Test_save(Test_data, 
                  mask_names,
                  unet, 
                  unet_opt, 
                  Param.Parameters.Network["Test_paths"]["Data_path"], 
                  Param.Parameters.Network["Test_paths"]["Extensions"], 
                  save_path, 
                  load_path, 
                  test_volume_list,
                  save=Save)

print("END")