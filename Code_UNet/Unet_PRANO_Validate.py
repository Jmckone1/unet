import Net_modules.Parameters_PRANO as Param
import torch.cuda.amp as amp
import numpy as np
import torch
from tqdm import tqdm

from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
from sklearn.metrics import jaccard_score

class UNet_validate():
    def __init__(self):
        print("Init")
        print(" ")
        print("Validation...")

def Validate(unet, criterion, Val_data):
    print(" ")
    print("Validation...")
    unet.eval()
    
    mse_values = []
    cosine_values = []
    losses = []
    
    running_loss = 0.0
    mse_run = 0.0
    cosine_run = 0.0
    
    cur_step = 0
    jaccard_val = []
    
    for truth_input, label_input in tqdm(Val_data):

        cur_batch_size = len(truth_input)

        # flatten ground truth and label masks
        truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()

        label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()
        
        pred = unet(truth_input)
        pred = pred.squeeze()

        # forward
        loss, mse, cosine = criterion(pred, label_input)
        # print("v loss", loss)
        
        loss.backward()
        
        running_loss =+ loss.item()
        mse_run =+ mse.item()
        cosine_run =+ cosine.item()
        # print("v run loss", running_loss)
        
        losses.append(running_loss)
        mse_values.append(mse_run)
        cosine_values.append(cosine_run)
        
        # print("v losses", losses)

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()

        for input_val in range(cur_batch_size):
            
            corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
            mask_truth = Jacc.mask((240,240),corners_truth)#*1
            corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
            mask_pred = Jacc.mask((240,240),corners_pred)#*1
            
            # print("Total sum of mask pixels", np.sum(np.sum(mask_truth)))
            if np.sum(np.sum(mask_truth)) > 2:
                jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
            else:
                jaccard_val.append(float("NaN"))
        
        cur_step += 1
    
    print("v j val",jaccard_val)
    print("Validation complete")
    print(" ")
    
    return losses, mse_values, cosine_values, jaccard_val

# def Validate(unet, criterion, Val_data):

#     unet.eval()
#     losses = []
    
#     running_loss = 0.0
#     mse_run = 0.0
#     cosine_run = 0.0
    

#     val_results = [[],[],[],[]]
    
#     cur_step = 0
#     jaccard_val = []
    
#     scaler = amp.GradScaler(enabled = True)
    
#     for truth_input, label_input in tqdm(Val_data):

#         cur_batch_size = len(truth_input)

#         # flatten ground truth and label masks
#         truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
#         truth_input = truth_input.float() 
#         truth_input = truth_input.squeeze()
#         label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
#         label_input = label_input.float()
#         label_input = label_input.squeeze()
        
#         truth_input = truth_input.to(dtype=torch.half)
#         label_input = label_input.to(dtype=torch.half)

#         with amp.autocast(enabled = True):
#             pred = unet(truth_input)
#             pred = pred.squeeze()

#         # forward
#         unet_loss, mse, cosine = criterion(pred, label_input)
#         scaler.scale(unet_loss).backward()
#         scaler.update()
        
#         running_loss =+ unet_loss.item()
#         mse_run =+ mse.item()
#         cosine_run =+ cosine.item() 
        
#         losses.append(running_loss / len(Val_data))

#         for input_val in range(cur_batch_size):
                
#             corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
#             mask_truth = Jacc.mask((240,240),corners_truth)*1
#             corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
#             mask_pred = Jacc.mask((240,240),corners_pred)*1

#             if np.sum(np.sum(mask_pred)) > 2:
#                 jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
#             else:
#                 jaccard_val.append(float("NaN"))
        
#         cur_step += 1
    
    
#     return running_loss, mse_run, cosine_run, jaccard_val