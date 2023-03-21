from Net_modules.Loading_data import Load_Dataset

from Net_modules.Evaluation import DiceLoss
from Net_modules.Penalty import Penalty

import Net_modules.Parameters_SEG as Param
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import Unet_FULL_Train

if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Train_seg.py start")

torch.cuda.empty_cache()

os.chdir(os.getcwd())

headers = ["Global","Hyperparameters","Train_paths"]
print("Torch version: ",torch.__version__)
print("")
for h in headers:
    for key, value in Param.Parameters.PRANO_Net[h].items():
        print(f'{key: <30}{str(value): <35}')
        
("Press Enter to load Global Parameters . . . ")    
    
np.set_printoptions(precision=4)
print("Loading seed . . .")
np.random.seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
torch.manual_seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Loaded Seed")

print("Loaded GPU allocation")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(Param.Parameters.PRANO_Net["Global"]["GPU"])

print("Loading loss . . .")
if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
    loss_f = DiceLoss()
    criterion = loss_f
else:
    loss_f = Penalty(Param.Parameters.PRANO_Net["Hyperparameters"]["Cosine_penalty"])
    criterion = loss_f.MSELossorthog

print("Loading Dataset")
Full_Path = os.getcwd() + "/" + Param.Parameters.PRANO_Net["Train_paths"]["Data_path"]
folder = np.loadtxt(Full_Path + "/Training_dataset.csv", delimiter=",",dtype=str)
image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

# folder = read_csv_paths
dataset = Load_Dataset(Full_Path,image_folder_in,masks_folder_in)
print("")

Dataset_size = len(folder)
print("Dataset size: ", Dataset_size)

split = folder[:,3].astype(int)

training_split = folder[(np.where(~np.logical_or(split==0, split==1))),2]
training_split = np.squeeze(training_split).astype(int)

validation_split = folder[(np.where(np.logical_or(split==0, split==1))),2]
validation_split = np.squeeze(validation_split).astype(int)

train_data = torch.utils.data.RandomSampler(training_split,False)
validation_data = torch.utils.data.RandomSampler(validation_split,False)

print("Full_dataset: ", len(split))
print("Training: ", len(training_split))
print("validation: ", len(validation_split))

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_size"],
    sampler=train_data)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_size"],
    sampler=validation_data)

if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Starting training . . .")
      
train_model = Unet_FULL_Train.UNet_train(criterion)
train_model.train(Train_data, Val_data, load=False)

if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Train_seg.py end")