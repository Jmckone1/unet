from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
import Net_modules.Parameters_SEG as Param
from sklearn.metrics import jaccard_score
import torch.cuda.amp as amp
from tqdm import tqdm
from torch import nn
import numpy as np
import shutil
import torch

class UNet_validate():
    def __init__(self):
        print("Init")
        print(" ")
        print("Validation...")

def Validate(unet, criterion, Val_data, epoch, step = ""):
    sigmoid_act = nn.Sigmoid()
#     print(" ")
#     print("Validation...")
    unet.eval()
    
    mse = []
    cosine = []
    loss = []
    
    running_loss = 0.0
    running_mse = 0.0
    running_cosine = 0.0
    improvement = 0
    
    cur_step = 0
    
    Dice, Jaccard = [], []
    
    for truth_input, label_input in tqdm(Val_data):
        
        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()
        label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
        if(truth_input.ndim == 3):
            truth_input = truth_input[:,np.newaxis,:,:]
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                label_input = label_input[:,np.newaxis,:,:]
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                label_input = label_input[:,np.newaxis,:]
        
        # set accumilated gradients to 0 for param update
#         with amp.autocast(enabled = True):
        pred = unet(truth_input)
        pred = pred.squeeze()
            
        if(pred.ndim == 2):
            pred = pred[np.newaxis,:,:]

        # forward
        unet_loss = criterion(pred,label_input)
        
        truth_output = label_input.cpu().detach().numpy().squeeze()

        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            # calculate jaccard score
            pred_output = pred.cpu().detach().numpy().squeeze()

            for Batch in range(cur_batch_size):
                corners_truth, center_truth = Jacc.Obb(truth_output[Batch,:])
                mask_truth = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"], corners_truth)
                corners_pred, center_pred = Jacc.Obb(pred_output[Batch,:])
                mask_pred = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],   corners_pred)

                if np.sum(np.sum(mask_pred)) > 2:
                    Jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    Jaccard.append(float("NaN"))

            unet_cosine = unet_loss[2]
            unet_mse = unet_loss[1]
            # reset unet loss to a non array structure
            unet_loss = unet_loss[0]
        else:
            #calculate dice score
            pred_output = sigmoid_act(pred).cpu().detach().numpy()
            for Batch in range(cur_batch_size):
                Dice.append(Dice_Eval.dice_score(pred_output[Batch,:,:],truth_output[Batch,:,:]))
            
        unet_loss.backward()
        
        running_loss =+ unet_loss.item()
        loss.append(running_loss)
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            running_mse =+ unet_mse.item()
            mse.append(running_mse)
            running_cosine =+ unet_cosine.item()
            cosine.append(running_cosine)
                
        cur_step += 1
    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
        return loss, mse, cosine, Jaccard
    else:
        return loss, Dice
    