# from Net_modules.Unet_Main_dataloader import BraTs_Dataset
from Net_modules.Loading_data import Load_Dataset
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
        
np.set_printoptions(precision=4)

input("Press Enter to continue . . . ")

if os.path.exists("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
    print("Path already exists")
    print("Please enter Y to delete file contents or anything else to exit: ")
    replace = input("")
    if replace == "Y":
        shutil.rmtree("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"])
        print("File deleted . . . continuing script")
    else:
        print("Exiting script")
        sys.exit()

criterion = nn.BCEWithLogitsLoss()

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def Validate(unet, criterion, Val_data, epoch, step = ""):

    print(" ")
    print("Validation...")
    
    unet.eval()
    
    running_loss = 0.0
    cur_step = 0
    
    for truth_input, label_input in tqdm(Val_data):
        DS, HD, Sp, Se = [],[],[],[]
        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
        truth_input = truth_input.float() 

        label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
        label_input = label_input.float()

        if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 4:
            truth_input = truth_input.squeeze()
            label_input = label_input.squeeze()
            if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
                truth_input = truth_input[np.newaxis,:,:,:]
                label_input = label_input[np.newaxis,:,:]
            label_input = label_input[:,np.newaxis,:,:]

        pred = unet(truth_input)

        pred = pred.squeeze()
        label_input = label_input.squeeze()
        if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:
            pred = pred[:,np.newaxis,:,:]
            label_input = label_input[:,np.newaxis,:,:]

        loss = criterion(pred, label_input)
        running_loss =+ loss.item()
        cur_step += 1
        
        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        if(pred_output.ndim == 2):
            pred_output = pred_output[np.newaxis,:,:]
            truth_output = truth_output[np.newaxis,:,:]
        
        for Batch in range(cur_batch_size):
            DS.append(Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:]))
            HD.append(100)
            Sp.append(1000)
            Se.append(10000)
        
        with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + step + "validation_loss.csv", 'a') as f: 
            np.savetxt(f, [running_loss], delimiter=',')
        with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + step + "validation_dice.csv", 'a') as f: 
            np.savetxt(f, [DS], delimiter=',')
            
        with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + step + "validation_metrics.csv", 'a') as f: 
            np.savetxt(f, [DS,HD,Sp,Se], delimiter=',')
                
    print("Validation complete")
    print(" ")

def train(Train_data,Val_data,load=False):
    
    if Param.Parameters.Network["Hyperparameters"]["Use_weights"] == True:
        unet = net.UNet.load_weights(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Hidden_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Regress"],
                                     Param.Parameters.Network["Hyperparameters"]["Allow_update"],
                                     Param.Parameters.Network["Train_paths"]["Checkpoint_load"]
                                     ).to(Param.Parameters.Network["Global"]["device"])
    else:
        unet = net.UNet(Param.Parameters.Network["Hyperparameters"]["Input_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                                     Param.Parameters.Network["Hyperparameters"]["Hidden_dim"]).to(Param.Parameters.Network["Global"]["device"])
        
    unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.Parameters.Network["Hyperparameters"]["Learning_rate"], weight_decay=Param.Parameters.Network["Hyperparameters"]["Weight_decay"])
    
    if not os.path.exists("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture"):

        os.makedirs("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture")
    
    with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture.txt", 'w') as write: 
        write.write("left_path: " + Param.Parameters.Network["Train_paths"]["Checkpoint_load"] + "\n")
        write.write("epochs: " + str(Param.Parameters.Network["Hyperparameters"]["Epochs"]) + "\n")
        write.write("batch size: " + str(Param.Parameters.Network["Hyperparameters"]["Batch_size"]) + "\n")
        write.write("learning rate: " + str(Param.Parameters.Network["Hyperparameters"]["Learning_rate"]) + "\n")
        write.write(str(unet))
        
    original = Param.Parameters.Network["Global"]["Param_location"]
    target = "Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_hyperparameters.py"
    if not os.path.exists("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
        os.makedirs("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"])
        
    shutil.copyfile(original, target)

    for epoch in range(Param.Parameters.Network["Hyperparameters"]["Epochs"]):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss = 0.0

        for truth_input, label_input in tqdm(Train_data, desc= running_loss):
            
            DS, HD, Sp, Se = [],[],[],[]
            
            cur_batch_size = len(truth_input)

            truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
            truth_input = truth_input.float() 

            label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
            label_input = label_input.float()

            if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 4:
                truth_input = truth_input.squeeze()
                label_input = label_input.squeeze()
#                 print(np.shape(label_input))
                if cur_batch_size == 1:
                    label_input = label_input[np.newaxis,np.newaxis,:,:]
                    truth_input = truth_input[np.newaxis,:,:,:]
                else:
                    label_input = label_input[:,np.newaxis,:,:]

                unet_opt.zero_grad()
            
                pred = unet(truth_input)
                pred = pred.squeeze()
                label_input = label_input.squeeze()
                
            if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:

                if cur_batch_size == 1:
                    pred = pred[np.newaxis,np.newaxis,:,:]
                    label_input = label_input[np.newaxis,np.newaxis,:,:]
                else:
                    pred = pred[:,np.newaxis,:,:]
                    label_input = label_input[:,np.newaxis,:,:]
                    
                unet_opt.zero_grad()
                
            unet_loss = criterion(pred, label_input)

            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item()
            cur_step += 1
            
            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            
            if Param.Parameters.Network["Global"]["Debug"] == True:
                for batchshow in range(cur_batch_size):
                    fig, ax = plt.subplots(2)
                    ax[0].imshow((pred_output[batchshow,0,:,:] > 0.5).astype(int), cmap='gray')
                    plt.title("Pred")
                    ax[1].imshow(truth_output[batchshow,0,:,:], cmap='gray')
                    plt.show()

                    print("Min/Max: ", np.min(pred_output[batchshow,0,:,:]),"/",np.max(pred_output[batchshow,0,:,:]))

            if(pred_output.ndim == 2):
                pred_output = pred_output[np.newaxis,:,:]
                truth_output = truth_output[np.newaxis,:,:]
            dice_val = 0
            for Batch in range(cur_batch_size):
                dice_val = Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:])
                if Param.Parameters.Network["Global"]["Debug"] == True:
                    print("Dice score:" , dice_val)
                
                DS.append(dice_val)
                HD.append(100)
                Sp.append(1000)
                Se.append(10000)
                
            with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + "training_loss.csv", 'a') as f: 
                np.savetxt(f, [running_loss], delimiter=',')
            with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + "training_dice.csv", 'a') as f: 
                # is there a reason that i am saving the mean instead of the raw values here and not for the loss? 
#                 np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                np.savetxt(f, [DS], delimiter=',')
            with open("Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "epoch_" + str(epoch) + "training_metrics.csv", 'a') as f: 
                # is there a reason that i am saving the mean instead of the raw values here and not for the loss? 
#                 np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                np.savetxt(f, [DS,HD,Sp,Se], delimiter=',')
            DS = []
            
        print("saving epoch: ", epoch)
        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
        out = "Checkpoints/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "checkpoint_" + str(epoch) + ".pth"
        torch.save(checkpoint, out)

        Validate(unet, criterion, Val_data, epoch)

    print('Finished Training Dataset')

print("Loading Dataset")
folder = np.loadtxt(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"] + "/Training_dataset.csv", delimiter=",",dtype=str)

image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

dataset = Load_Dataset(os.getcwd() + Param.Parameters.Network["Train_paths"]["Data_path"],
                       image_folder_in,
                       masks_folder_in,
                       Param.Parameters.Network["Hyperparameters"]["Apply_Augmentation"])

Dataset_size = len(folder)

print("Total number: ", Dataset_size)
print("Non empty masks: ",len(folder[np.where(folder[:,-1].astype(float) > 0)]))
print("Empty masks: ",len(folder[np.where(folder[:,-1].astype(float) == 0)]))

# training_split = folder[np.where(folder[:,3].astype(int) < 1),2]
# # training_split = folder[np.where(folder[:,3].astype(int) < 2),2]
# # training_split = folder[np.where(folder[:,3].astype(int) < 3),2]
# # training_split = folder[np.where(folder[:,3].astype(int) < 4),2]
# # training_split = folder[np.where(folder[:,3].astype(int) < 5),2]
# # training_split = folder[np.where(folder[:,3].astype(int) < 9),2]
# training_split = np.squeeze(training_split).astype(int)

# validation_split = folder[np.where(folder[:,3].astype(int) == 8),2]
# validation_split = np.squeeze(validation_split).astype(int)

# test_split = folder[np.where(folder[:,3].astype(int) == 9),2]
# test_split = np.squeeze(test_split).astype(int)

# train_data = torch.utils.data.RandomSampler(training_split,False)
# validation_data = torch.utils.data.RandomSampler(validation_split,False)

data_splits = [np.array([]), np.array([]), np.array([])]
data_split_values = [Param.Parameters.Network["Hyperparameters"]["Train_split"],
                     Param.Parameters.Network["Hyperparameters"]["Validation_split"],
                     Param.Parameters.Network["Hyperparameters"]["Test_split"]]

for splits in range(len(data_splits)):
    for val in data_split_values[splits]:
        if np.size(data_splits[splits]) == 0:
            data_splits[splits] = folder[np.where(folder[:,3].astype(int) == val),2]
        else:
            if np.size(folder[np.where(folder[:,3].astype(int) == val),2]) != 0:
                data_splits[splits] = np.concatenate((data_splits[splits],
                                                      folder[np.where(folder[:,3].astype(int) == val),2]), 
                                                      axis=1)

data_splits[0] = np.squeeze(data_splits[0])
data_splits[1] = np.squeeze(data_splits[1])
data_splits[2] = np.squeeze(data_splits[2])

train_data = torch.utils.data.RandomSampler(data_splits[0],False)
validation_data = torch.utils.data.RandomSampler(data_splits[1],False)

# print(np.shape(data_splits[0]),np.shape(data_splits[1]),np.shape(data_splits[2]))

# input("PAUSED")

del folder

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    num_workers=0,
    sampler=train_data, 
    pin_memory=True,
    worker_init_fn=_init_fn)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    num_workers=0,
    sampler=validation_data, 
    pin_memory=True,
    worker_init_fn=_init_fn)

print("")
print("Actual train length", len(Train_data.sampler))
print("actual validation length", len(Val_data.sampler))

train(Train_data, Val_data, load=False)