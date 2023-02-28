import Net_modules.Parameters_SEG as Param
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

def Validate(unet, criterion, Val_data, epoch, step = ""):
#     print(" ")
#     print("Validation...")
    unet.eval()
    
    mse_values = []
    cosine_values = []
    losses = []
    
    running_loss = 0.0
    mse_run = 0.0
    cosine_run = 0.0
    improvement = 0
    
    cur_step = 0
    
    DS, jaccard_val = [], []
    
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
        with amp.autocast(enabled = True):
            pred = unet(truth_input)
            pred = pred.squeeze()
            
            if(pred.ndim == 2):
                pred = pred[np.newaxis,:,:]

        # forward
        # loss, mse, cosine = criterion(pred, label_input)
        unet_loss = criterion(pred,label_input)
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            unet_cosine = unet_loss[2]
            unet_mse = unet_loss[1]
            unet_loss = unet_loss[0]
            
        unet_loss.backward()
        
        running_loss =+ unet_loss.item()
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            running_mse =+ unet_mse.item()
            running_cosine =+ uent_cosine.item()
            pred_output = pred.cpu().detach().numpy()
        else:
            pred_output = sigmoid_act(pred).cpu().detach().numpy()

        truth_output = label_input.cpu().detach().numpy()
        
        # these lines seem counterproductive so i shall see if both are necessary or whether the newaxis line should be removed.
        if(pred_output.ndim == 2):
            pred_output = pred_output[np.newaxis,:,:]
            truth_output = truth_output[np.newaxis,:,:]

        pred_output = np.squeeze(pred_output)
        truth_output = np.squeeze(truth_output)
        ##################################################################
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
            for i in range(cur_batch_size):
                DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))

            with open("Checkpoints/" + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                      "epoch_" + str(epoch) + step + "validation_loss.csv", 'a') as f: 
                np.savetxt(f, [running_loss], delimiter=',')
            with open("Checkpoints/" + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                      "epoch_" + str(epoch) + step + "validation_dice.csv", 'a') as f: 
                np.savetxt(f, [np.nanmean(DS)], delimiter=',')

        ###################################################################
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
            for input_val in range(cur_batch_size):

                corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
                mask_truth = Jacc.mask((240,240),corners_truth)#*1
                corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
                mask_pred = Jacc.mask((240,240),corners_pred)#*1
                if np.sum(np.sum(mask_truth)) > 2:
                    jaccard_val.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                else:
                    jaccard_val.append(float("NaN"))
# this section may need to fit in the training code instead of here so make sure to think and then move it .
############################################################################################################
            print("Improvement", Improvement)
            print("")

            # save a checkpoint only if there has been an improvement in the total jaccard score for the model.
            if np.nanmean(jaccard_val) > Improvement:
                if np.nanmean(epoch_jaccard_valid) == np.isnan:
                    Improvement = 0
                else:
                    Improvement = np.nanmean(epoch_jaccard_valid)

                print("saving epoch: ", epoch)
                checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                out = "Checkpoints_RANO/" + Param.rNet.checkpoint + "checkpoint_" + str(epoch) + ".pth"
                torch.save(checkpoint, out)

            with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_loss/epoch_" + str(epoch) + "training_loss.csv", 'w') as f: 
                write = csv.writer(f) 
                write.writerow(loss_values)

            with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_loss/epoch_" + str(epoch) + "validation_loss.csv", 'w') as f: 
                write = csv.writer(f) 
                write.writerow(valid_loss)

            with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Training_Jaccard/epoch_" + str(epoch) + "jaccard_index.csv", 'w') as f: 
                write = csv.writer(f) 
                write.writerow(jaccard)

            with open("Checkpoints_RANO/" + Param.rNet.checkpoint + "Validation_Jaccard/epoch_" + str(epoch) + "validation_jaccard_index.csv", 'w') as f: 
                write = csv.writer(f) 
                write.writerow(valid_jaccard)
        
        cur_step += 1
    
    print("v j val",jaccard_val)
    print("Validation complete")
    print(" ")
    
    return losses, mse_values, cosine_values, jaccard_val