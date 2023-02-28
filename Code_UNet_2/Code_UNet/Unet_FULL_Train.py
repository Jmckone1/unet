from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
import Net_modules.Parameters_SEG as Param
from Unet_PRANO_Validate import Validate

import Net.pytorch_dcsaunet.DCSAU_Net as net

import torch.cuda.amp as amp
from tqdm import tqdm
from torch import nn
import numpy as np
import shutil
import torch
import os

sigmoid_act = nn.Sigmoid()

class UNet_train(): 
    def __init__(self, criterion):
        print("Initilising Network . . .")
        
        self.criterion = criterion
        
        output_types = ["","Training_loss","Training_loss_mse","Training_loss_cosine",
                        "Validation_loss","Validation_loss_mse","Validation_loss_cosine",
                        "Training_Jaccard","Validation_Jaccard"]
        if not os.path.exists("Checkpoints_RANO/" + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"]):
            for name in output_types:
                os.makedirs("Checkpoints_RANO/" + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + name)
            
            original = r'Code_UNet/Net_modules/Parameters_PRANO.py'
            target = r'Checkpoints_RANO/' + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 'Parameters.py'
            shutil.copyfile(original, target)
            
    def train(self,Train_datas, Val_data,load=False):

        scaler = amp.GradScaler(enabled = True)
        
        # regress swaps the model between regression and the segmentation model
        unet = net.Model(4,1,Regress = Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"]).to(
                            Param.Parameters.PRANO_Net["Global"]["device"])
        
        print(unet)
        with open("Checkpoints_RANO/" + 
                  Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                  "Model_architecture", 'w') as write: 
            write.write(str(unet))
            
        unet_opt = torch.optim.Adam(unet.parameters(), 
                                    lr=Param.Parameters.PRANO_Net["Hyperparameters"]["Learning_rate"], 
                                    betas=Param.Parameters.PRANO_Net["Hyperparameters"]["Betas"],
                                    weight_decay=Param.Parameters.PRANO_Net["Hyperparameters"]["Weight_decay"])
    
        if load == True:
            checkpoint = torch.load("Checkpoints_RANO/" + 
                                    Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                                    "checkpoint_0_step_1900.pth")
    
            unet.load_state_dict(checkpoint['state_dict'])
            unet_opt.load_state_dict(checkpoint['optimizer'])

        for epoch in range(Param.Parameters.PRANO_Net["Hyperparameters"]["Epochs"]):
            cur_step = 0
            
            print("Training...")
            if epoch == 0 and load == True:
                epoch = checkpoint['epoch'] + 1
                
            unet.train()
            running_loss = 0.0
            DS, jaccard = [], []
            
            for truth_input, label_input in tqdm(Train_datas):
                
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
                unet_opt.zero_grad(set_to_none=True)
                with amp.autocast(enabled = True):
                    pred = unet(truth_input)
                    pred = pred.squeeze()
                
                    if(pred.ndim == 2):
                        pred = pred[np.newaxis,:,:]
                
                    # forward
                    unet_loss = self.criterion(pred, label_input)
                
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    unet_cosine = unet_loss[2]
                    unet_mse = unet_loss[1]
                    unet_loss = unet_loss[0]
                    
                scaler.scale(unet_loss).backward()
                scaler.step(unet_opt)
                scaler.update()

                running_loss =+ unet_loss.item()
                
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    running_mse =+ unet_mse.item()
                    running_cosine =+ uent_cosine.item()
                    pred_output = pred.cpu().detach().numpy()
                else:
                    pred_output = sigmoid_act(pred).cpu().detach().numpy()
                    
                truth_output = label_input.cpu().detach().numpy()
                
                
                # edgecase handling in regard to a single image being left in a batch at the end of training. otherwise causing a crash before evaluation.
                if(pred_output.ndim == 2):
                    pred_output = pred_output[np.newaxis,:,:]
                    truth_output = truth_output[np.newaxis,:,:]
                    
                pred_output = np.squeeze(pred_output)
                truth_output = np.squeeze(truth_output)
                
                ##################################################################
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                    for i in range(cur_batch_size):
                        DS.append(Dice_Eval.dice_score(pred_output[i,:,:],truth_output[i,:,:]))
                        
                    with open(Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "_Epoch_" + str(epoch) + "_Training_Loss.csv", 'a') as f: 
                        np.savetxt(f, [running_loss], delimiter=',')
                    with open(Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "_Epoch_" + str(epoch) + "_Training_Dice.csv", 'a') as f: 
                        np.savetxt(f, [np.nanmean(DS)], delimiter=',')
                        
                ###################################################################
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    for i in range(cur_batch_size):
                        corners_truth, center_truth = Jacc.Obb(truth_output[i,:])
                        mask_truth = Jacc.mask((240,240),corners_truth)*1
                        corners_pred, center_pred = Jacc.Obb(pred_output[i,:])
                        mask_pred = Jacc.mask((240,240),corners_pred)*1

                        if np.sum(np.sum(mask_pred)) > 2:
                            jaccard.append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                        else:
                            jaccard.append(float("NaN"))
                            
                cur_step += 1        
                ###################################################################
                
                if cur_step % Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_display_step"] == 0:
                    if epoch == 0 and cur_step <= 250:
    
                        checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                        out = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "_Checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                        torch.save(checkpoint, out)
    
                        # if enabled this will validate the model during every *DISPLAY STEP* (default 50 batches) during the first epoch
    
                        if Param.Parameters.PRANO_Net["Hyperparameters"]["Evaluate"] == True:
                            if epoch == 0:
                                Validate(unet, self.criterion, Val_data, epoch, step = "_" + str(cur_step))
    
            print("saving epoch: ", epoch)
            checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
            out = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "_Checkpoint_" + str(epoch) + ".pth"
            torch.save(checkpoint, out)
    
            Validate(unet, self.criterion, Val_data, epoch)
    
        print('Finished Training Dataset')