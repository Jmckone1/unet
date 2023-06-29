from Net_modules.Loading_data import Load_Dataset
import Net_modules.Model_hyperparameters as Param
import Net.Unet_components_split as net

from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
from Net_modules.Penalty import Penalty

from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import numpy as np
import logging
import shutil
import torch
import csv
import os
import random

seed = Param.Parameters.Network["Global"]["Seed"]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = Param.Parameters.Network["Global"]["Enable_Determinism"]
torch.backends.cudnn.enabled = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(Param.Parameters.Network["Global"]["GPU"])

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

if os.path.exists("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
    print("Path already exists")
    print("Please enter Y to delete file contents or anything else to exit: ")
    replace = input("")
    if replace == "Y":
        shutil.rmtree("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"])
        print("File deleted . . . continuing script")
    else:
        print("Exiting script")
        sys.exit()

np.set_printoptions(precision=4)

loss_f = Penalty(Param.Parameters.Network["Hyperparameters"]["Cosine_penalty"])
criterion = loss_f.MSELossorthog

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

### added a bunch of outputs for here to test when i get back!!!
def Validate(unet, criterion, Val_data):
    
    np.random.seed(Param.Parameters.Network["Global"]["Seed"])
    torch.manual_seed(Param.Parameters.Network["Global"]["Seed"])
    print(" ")
    print("Validation...")
    unet.eval()
    
    mse_values, cosine_values, losses = [],[],[]
    running_loss, mse_run, cosine_run = 0.0, 0.0, 0.0
    
    cur_step = 0
    jaccard_val = []
    
    for truth_input, label_input in tqdm(Val_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
        if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
            # print(np.shape(truth_input), np.shape(label_input))
            truth_input = truth_input[np.newaxis,:,:,:]
            label_input = label_input[np.newaxis,:]
#                 print(np.shape(truth_input), np.shape(label_input))

        if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:
#                 print("Input dim start")
            truth_input = truth_input.squeeze()
            label_input = label_input.squeeze()
            if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
                truth_input = truth_input[np.newaxis,:,:,:]
                label_input = label_input[np.newaxis,:]
#                 print("Label change")
            label_input = label_input[:,np.newaxis,:]
            truth_input = truth_input[:,np.newaxis,:,:]

        label_input = label_input.squeeze()
            
        print(np.shape(truth_input))
        print(np.shape(label_input))

        pred = unet(truth_input)
        pred = pred.squeeze()

        # forward
        loss, mse, cosine = criterion(pred, label_input)

        loss.backward()
        
        running_loss =+ loss.item()
        mse_run =+ mse.item()
        cosine_run =+ cosine.item()

        losses.append(running_loss)
        mse_values.append(mse_run)
        cosine_values.append(cosine_run)

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        for input_val in range(cur_batch_size):
            
            corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
            mask_truth = Jacc.mask((240,240),corners_truth)#*1
            corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
            mask_pred = Jacc.mask((240,240),corners_pred)#*1

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
    
    np.random.seed(Param.Parameters.Network["Global"]["Seed"])
    torch.manual_seed(Param.Parameters.Network["Global"]["Seed"])
    Improvement = 0
    
    unet = net.UNet(Param.Parameters.Network["Hyperparameters"]["Input_dim"],
                    Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                    Param.Parameters.Network["Hyperparameters"]["Hidden_dim"]).to(
        Param.Parameters.Network["Global"]["device"])
    print(unet)
    
    Files = ["","Training_loss","Training_loss_mse",
            "Training_loss_cosine","Validation_loss",
            "Validation_loss_mse","Validation_loss_cosine",
            "Training_Jaccard","Validation_Jaccard"]
    
    if not os.path.exists("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
        for file in Files: 
            os.makedirs("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + file)
    
    ## may get rid of this section in the future as we now save the parameter file, though the architecture is potentially useful.
    with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_architecture", 'w') as write: 
        write.write("epochs: " + str(Param.Parameters.Network["Hyperparameters"]["Epochs"]) + "\n")
        write.write("batch size: " + str(Param.Parameters.Network["Hyperparameters"]["Batch_size"]) + "\n")
        write.write("learning rate: " + str(Param.Parameters.Network["Hyperparameters"]["Learning_rate"]) + "\n")
        write.write("orthogonality weight: " + str(Param.Parameters.Network["Hyperparameters"]["Cosine_penalty"]) + "\n")
        write.write(str(unet))

    original = Param.Parameters.Network["Global"]["Param_location"]
    target = "Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Model_hyperparameters.py"
    if not os.path.exists("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"]):
        os.makedirs("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"])

    unet_opt = torch.optim.Adam(unet.parameters(), 
                                Param.Parameters.Network["Hyperparameters"]["Learning_rate"],
                                betas=Param.Parameters.Network["Hyperparameters"]["Betas"], weight_decay=Param.Parameters.Network["Hyperparameters"]["Weight_decay"])

    if load == True:
        checkpoint = torch.load("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_load"])

        unet.load_state_dict(checkpoint['state_dict'])
        unet_opt.load_state_dict(checkpoint['optimizer'])

#                   Define model end                     #
#--------------------------------------------------------#
#                   Run model start                      #
    t, v, total_loss = [],[],[]
    scaler = amp.GradScaler(enabled = True)

    for epoch in range(Param.Parameters.Network["Hyperparameters"]["Epochs"]):
        cur_step = 0
        
        print("Training...")
        if epoch == 0 and load == True:
            epoch = checkpoint['epoch'] + 1
            
        unet.train()
        
        running_loss, mse_run, cosine_run= 0.0, 0.0, 0.0
        loss_values,mse_values,cosine_values = [],[],[]
        valid_loss,valid_mse_values,valid_cosine_values = [],[],[]
        jaccard,valid_jaccard = [],[]
        
        for truth_input, label_input in tqdm(Train_data):

            cur_batch_size = len(truth_input)
            
            # flatten ground truth and label masks
            truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
            label_input = label_input.float()
            label_input = label_input.squeeze()
            
#             print(np.shape(truth_input), np.shape(label_input))
            if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
                # print(np.shape(truth_input), np.shape(label_input))
                truth_input = truth_input[np.newaxis,:,:,:]
                label_input = label_input[np.newaxis,:]
#                 print(np.shape(truth_input), np.shape(label_input))

            if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:
#                 print("Input dim start")
                truth_input = truth_input.squeeze()
                label_input = label_input.squeeze()
                if cur_batch_size == 1: #!= Param.Parameters.Network["Hyperparameters"]["Batch_size"]:
                    truth_input = truth_input[np.newaxis,:,:,:]
                    label_input = label_input[np.newaxis,:]
#                 print("Label change")
                label_input = label_input[:,np.newaxis,:]
                truth_input = truth_input[:,np.newaxis,:,:]
                
            label_input = label_input.squeeze()
            
#             print(np.shape(truth_input),np.shape(label_input))

            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()

            pred = unet(truth_input)
#             print("1", np.shape(pred))
#             pred = pred.squeeze()

            # forward
            
#             print(np.shape(pred),np.shape(label_input))
#             input("")
            
#             print("2", np.shape(pred), np.shape(label_input))
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
            
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item()
            cur_step += 1
            
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
        
            if cur_step % Param.Parameters.Network["Hyperparameters"]["Batch_display_step"] == 0:

#                 print("Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
#                 print(label_input[0,:].shape)

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
            out = "Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "checkpoint_" + str(epoch) + ".pth"
            torch.save(checkpoint, out)
            
        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Training_loss/epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(loss_values)
            
        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Training_loss_mse/epoch_" + str(epoch) + "training_loss_mse.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(mse_values)
            
        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Training_loss_cosine/epoch_" + str(epoch) + "training_loss_cosine.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(cosine_values)

        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Validation_loss/epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_loss)
            
        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Validation_loss_mse/epoch_" + str(epoch) + "validation_loss_mse.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_mse_values)
            
        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Validation_loss_cosine/epoch_" + str(epoch) + "validation_loss_cosine.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_cosine_values)

        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Training_Jaccard/epoch_" + str(epoch) + "jaccard_index.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(jaccard)

        with open("Checkpoints_RANO/" + Param.Parameters.Network["Train_paths"]["Checkpoint_save"] + "Validation_Jaccard/epoch_" + str(epoch) + "validation_jaccard_index.csv", 'w') as f: 
            write = csv.writer(f) 
            write.writerow(valid_jaccard)

    print('Finished Training Dataset')
    return total_loss, valid_loss

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

training_split = folder[np.where(folder[:,3].astype(int) < 9),2]
# training_split = folder[np.where(folder[:,3].astype(int) < 2),2]
# training_split = folder[np.where(folder[:,3].astype(int) < 3),2]
# training_split = folder[np.where(folder[:,3].astype(int) < 4),2]
# training_split = folder[np.where(folder[:,3].astype(int) < 5),2]
# training_split = folder[np.where(folder[:,3].astype(int) < 9),2]
training_split = np.squeeze(training_split).astype(int)

validation_split = folder[np.where(folder[:,3].astype(int) == 9),2]
validation_split = np.squeeze(validation_split).astype(int)

test_split = folder[np.where(folder[:,3].astype(int) == 2),2]
test_split = np.squeeze(test_split).astype(int)

train_data = torch.utils.data.RandomSampler(training_split,False)
validation_data = torch.utils.data.RandomSampler(validation_split,False)
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