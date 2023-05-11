from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
import Net_modules.Parameters_SEG as Param

from sklearn.metrics import jaccard_score
import torch.cuda.amp as amp
from tqdm.auto import tqdm
from torch import nn
import numpy as np
import shutil
import torch

class UNet_validate():
    def __init__(self):
        # torch.cuda.empty_cache()
        print("Init")
        print(" ")
        print("Validation...")

def Validate(unet, criterion, Val_data, epoch, step = ""):
    
    unet.eval()
    Debug = Param.Parameters.PRANO_Net["Global"]["Debug"]
    
    sigmoid_act = nn.Sigmoid()
    
    # mse, cosine, loss = [],[],[]
    # Dice, Jaccard = [], []
    
    running_loss, running_mse, running_cosine = 0.0, 0.0, 0.0
    
    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
        # loss, mse, cosine, jaccard
        val_results = [[],[],[],[]]
    else:
        # loss, dice
        val_results = [[],[]]

    for truth_input, label_input in tqdm(Val_data):
        
        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()
        label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
#         truth_input = truth_input.to(dtype=torch.half)
#         label_input = label_input.to(dtype=torch.half)
        
        if(truth_input.ndim == 3):
            truth_input = truth_input[:,np.newaxis,:,:]
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                label_input = label_input[:,np.newaxis,:,:]
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                label_input = label_input[:,np.newaxis,:]
        
        # set accumilated gradients to 0 for param update
        pred = unet(truth_input)
        pred = pred.squeeze()

        if pred.ndim == 1:
            pred = pred[np.newaxis,:]
        if pred.ndim == 2:
            pred = pred[np.newaxis,:,:]
        # forward
        unet_loss = criterion(pred,label_input)
        
        truth_output = label_input.cpu().detach().numpy().squeeze()

        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            
            pred_output = pred.cpu().detach().numpy().squeeze()

            if Debug: print("Pred Magenta, Truth Yellow")
            # calculate jaccard score
            for Batch in range(cur_batch_size):
                if Debug:
                    print("input", truth_output[Batch,:])
                    print("prediction", pred_output[Batch,:])
                corners_truth, center_truth = Jacc.Obb(truth_output[Batch,:])
                mask_truth = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"], corners_truth)
                corners_pred, center_pred = Jacc.Obb(pred_output[Batch,:])
                mask_pred = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],   corners_pred)

                if np.sum(np.sum(mask_pred)) > 2:
                    val_results[1].append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))

                else:
                    val_results[1].append(float(0)) #.append(float("NaN"))
                    
                if Debug: print("Jaccard score: ", val_results[1][-1]) 
                    
            if Debug:
                backdrop = np.zeros((Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][0],
                                    Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"][1]))
                fig = plt.figure(figsize=(10,6))
                grid = ImageGrid(fig, 111,nrows_ncols=(2, 4),axes_pad=0.1)

                for ax, im in zip(grid, truth_output):
                    ax.imshow(backdrop,cmap='gray')
                    D1 = np.asarray([[im[1],im[3]],
                                     [im[0],im[2]]]) 
                    D2 = np.asarray([[im[5],im[7]],
                                     [im[4],im[6]]]) 

                    ax.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
                    ax.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')

                for ax, im in zip(grid, pred_output):                    
                    D3 = np.asarray([[im[1],im[3]],
                                     [im[0],im[2]]]) 
                    D4 = np.asarray([[im[5],im[7]],
                                     [im[4],im[6]]]) 

                    ax.plot(D3[0, :], D3[1, :], lw=2, c='m',label='_nolegend_')
                    ax.plot(D4[0, :], D4[1, :], lw=2, c='m',label='Prediction')

                plt.show()
                    
            unet_cosine = unet_loss[2]
            unet_mse = unet_loss[1]
            # reset unet loss to a non array structure
            unet_loss = unet_loss[0]
            if Debug:
                print("validation loss", unet_loss)
                print("validation jaccard", val_results[1][-1])
                print("validation cosine", unet_cosine)
                print("validation mse", unet_mse[-1])
        else:
            #calculate dice score
            
            pred_output = sigmoid_act(pred).cpu().detach().numpy()
            for Batch in range(cur_batch_size):
                if Debug: 
                    print("DICE SCORE: ", 
                          Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int), 
                                                          truth_output[Batch,:,:]), 
                          "PRIOR SCORE" ,
                          Dice_Eval.dice_score(pred_output[Batch,:,:],
                                               truth_output[Batch,:,:]))
                    
                val_results[1].append(Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:]))
            
#         unet_loss.backward()
        
        running_loss =+ unet_loss.item()
        val_results[0].append(running_loss)
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            running_mse =+ unet_mse.item()
            running_cosine =+ unet_cosine.item()
            
            val_results[2].append(running_mse)
            val_results[3].append(running_cosine)

    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
        return val_results[0], val_results[1], val_results[2], val_results[3]
    else:
        return val_results[0], val_results[1]
    