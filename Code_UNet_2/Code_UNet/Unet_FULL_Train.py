from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc

import Net.pytorch_dcsaunet.DCSAU_Net as net
#import Net.UNet_components as net

import Net_modules.Parameters_SEG as Param
from Unet_FULL_Validate import Validate
from sklearn.metrics import jaccard_score
import torch.cuda.amp as amp
from tqdm.auto import tqdm
from torch import nn
import numpy as np
import shutil
import torch
import csv
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class UNet_train(): 
    def __init__(self, criterion):
        print("Initilising Network . . .")
        torch.cuda.empty_cache()
        
        self.Debug = Param.Parameters.PRANO_Net["Global"]["Debug"]
        
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"]: self.model_type = "Regression"
        else: self.model_type = "Segmentation"
        
        self.criterion = criterion
        
        output_types = ["","Training_loss","Training_loss_mse","Training_loss_cosine",
                        "Validation_loss","Validation_loss_mse","Validation_loss_cosine",
                        "Training_Jaccard","Validation_Jaccard"]
        
        if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Checking if following path exists: ",
                                                                        Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"])
            
        self.Total_path = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "/" + self.model_type + "/"
        if os.path.exists(self.Total_path) == False:
            os.makedirs(self.Total_path)
            os.makedirs(self.Total_path + "Checkpoints/")
        
        print("Starting Param file copy . . .")
        original = Param.Parameters.PRANO_Net["Global"]["Param_location"]
        target = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "/" + self.model_type + "/Parameters.py"
        shutil.copyfile(original, target)
        print("Param file copy complete . . .")
        if Param.Parameters.PRANO_Net["Global"]["Debug"] == True: print("Copied file saved at location: ",
                                                                        Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] 
                                                                        + "/Parameters.py")

        print("################")
        print("#", self.model_type, "#")
        print("################")

        
            
    def train(self,Train_datas, Val_data,load=False):
        
        sigmoid_act = nn.Sigmoid()
        scaler = amp.GradScaler(enabled = True)
        
        # regress swaps the model between regression and the segmentation model
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Use_weights"] == True:
            unet = net.Model.load_weights(1,
                                          1,
                                          Regress = Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"],
                            Allow_update = Param.Parameters.PRANO_Net["Hyperparameters"]["Allow_update"],
                            Checkpoint_name = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_load"]).to(
                            Param.Parameters.PRANO_Net["Global"]["device"])
        else:
            unet = net.Model(1,1,Regress = Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"]).to(
                            Param.Parameters.PRANO_Net["Global"]["device"])
            
        if self.Debug: print("Created model", torch.cuda.memory_allocated()/1024**2)
        
        print(unet)
        with open( self.Total_path + "/" + "Model_architecture", "w") as write: 
            write.write(str(unet))
            
        unet_opt = torch.optim.Adam(unet.parameters(), 
                                    lr=Param.Parameters.PRANO_Net["Hyperparameters"]["Learning_rate"], 
                                    betas=Param.Parameters.PRANO_Net["Hyperparameters"]["Betas"],
                                    weight_decay=Param.Parameters.PRANO_Net["Hyperparameters"]["Weight_decay"])
    
        if load == True:
            checkpoint = torch.load("Checkpoints_RANO/" + 
                                    Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                                    "checkpoint_0_step_1900.pth")
    
            unet.load_state_dict(checkpoint["state_dict"])
            unet_opt.load_state_dict(checkpoint["optimizer"])

        for epoch in range(Param.Parameters.PRANO_Net["Hyperparameters"]["Epochs"]):
            cur_step = 0
            
            print("Training...")
            if epoch == 0 and load == True:
                epoch = checkpoint["epoch"] + 1
                
            unet.train()
            
            running_loss, running_mse, running_cosine = 0.0, 0.0, 0.0
            
            # loss, mse, cosine, jaccard
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                # loss, mse, cosine, jaccard
                train_results = [[],[],[],[]]
                val_results = [[],[],[],[]]
                
                list_of_names = [["_loss/","epoch_" + str(epoch) + "_loss.csv"],
                             ["_loss_mse/","epoch_" + str(epoch) + "_loss_mse.csv"],
                             ["_loss_cosine/","epoch_" + str(epoch) + "_loss_cosine.csv"],
                             ["_Jaccard/","epoch_" + str(epoch) + "_jaccard_index.csv"]]
            else:
                # loss, dice
                train_results = [[],[]]
                val_results = [[],[]]
                
                list_of_names = [["_loss/","epoch_" + str(epoch) + "_loss.csv"],
                             ["_Dice/","epoch_" + str(epoch) + "_dice_score.csv"]]

            Improvement = 0
            
            for truth_input, label_input in tqdm(Train_datas):
                
                cur_batch_size = len(truth_input)
                
                # flatten ground truth and label masks
                truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
                truth_input = truth_input.float() 
                truth_input = truth_input.squeeze()
                if self.Debug: print("Input image data", torch.cuda.memory_allocated()/1024**2)
                label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
                label_input = label_input.float()
                label_input = label_input.squeeze()
                if self.Debug: print("Input label data", torch.cuda.memory_allocated()/1024**2)
                
                if(truth_input.ndim == 3):
                    truth_input = truth_input[:,np.newaxis,:,:]
                    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                        label_input = label_input[:,np.newaxis,:,:]
                    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                        label_input = label_input[:,np.newaxis,:]
                
                
                # set accumilated gradients to 0 for param update
                unet_opt.zero_grad(set_to_none=True)
                if self.Debug: print("zero grad", torch.cuda.memory_allocated()/1024**2)
                with amp.autocast(enabled = True):
                    pred = unet(truth_input)
                    pred = pred.squeeze()
                    if self.Debug: print("autocast pred and prediction", torch.cuda.memory_allocated()/1024**2)
                    if pred.ndim == 1:
                        pred = pred[np.newaxis,:]
                    if pred.ndim == 2:
                        pred = pred[np.newaxis,:,:]

                    # forward
                    unet_loss = self.criterion(pred, label_input)
                    
                truth_output = label_input.cpu().detach().numpy().squeeze()
                
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    # calculate jaccard score
                    pred_output = pred.cpu().detach().numpy().squeeze()
                    
                    for Batch in range(cur_batch_size):
                        corners_truth, center_truth = Jacc.Obb(truth_output[Batch,:])
                        mask_truth = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],corners_truth)
                        corners_pred, center_pred = Jacc.Obb(pred_output[Batch,:])
                        mask_pred = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],corners_pred)

                        if np.sum(np.sum(mask_pred)) > 2:
                            train_results[3].append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                        else:
                            train_results[3].append(float("NaN"))
                            
                    unet_cosine = unet_loss[2]
                    unet_mse = unet_loss[1]
                    # reset unet loss to a non array structure
                    unet_loss = unet_loss[0]
                else:
                    #calculate dice score
                    
                    pred_output = sigmoid_act(pred).cpu().detach().numpy()
                    for Batch in range(cur_batch_size):
                        if self.Debug: print("DICE SCORE: ", Dice_Eval.dice_score((pred_output[Batch,:,:] < 0.5).astype(int),truth_output[Batch,:,:]))
                        train_results[1].append(Dice_Eval.dice_score((pred_output[Batch,:,:] < 0.5).astype(int),truth_output[Batch,:,:]))
                    if self.Debug:
                        fig = plt.figure()
                        grid = ImageGrid(fig, 111,nrows_ncols=(2, 4),axes_pad=0.1)

                        for ax, im in zip(grid, truth_output):
                            ax.imshow(im)
                        print("Truth_output")
                        plt.show()   

                        fig2 = plt.figure()
                        grid2 = ImageGrid(fig2,111,nrows_ncols=(2, 4),axes_pad=0.1)

                        for ax2, im2 in zip(grid2, (pred_output < 0.5).astype(int)):
                            ax2.imshow(im2)
                        print("Pred_output")
                        plt.show()    
                        
                if self.Debug: print("Just before backwards", torch.cuda.memory_allocated()/1024**2)
                scaler.scale(unet_loss).backward()
                scaler.step(unet_opt)
                scaler.update()

                running_loss =+ unet_loss.item()
                train_results[0].append(running_loss)
                
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    running_mse =+ unet_mse.item()
                    running_cosine =+ unet_cosine.item()
                    
                    train_results[1].append(running_mse)
                    train_results[2].append(running_cosine)
                            
                cur_step += 1        
                ###################################################################

                if cur_step % Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_display_step"] == 0:
                    if epoch == 0 and cur_step <= 250:
    
                        checkpoint = {"epoch": epoch,
                                      "state_dict": unet.state_dict(), 
                                      "optimizer" : unet_opt.state_dict()}
                        out = self.Total_path + "Checkpoints/" + "checkpoint_" + str(epoch) + "_" + str(cur_step) + ".pth"
                        torch.save(checkpoint, out)
    
#                         # if enabled this will validate the model during every *DISPLAY STEP* (default 50 batches) during the first epoch
#                         if Param.Parameters.PRANO_Net["Hyperparameters"]["Evaluate"] == True:
#                             if epoch == 0:
#                                 Validate(unet, self.criterion, Val_data, epoch, step = "_" + str(cur_step))
#                         # this section would need to include the saving for the intermediate output results
    
            print("Validation...")
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                val_loss, val_dice = Validate(unet, self.criterion,Val_data,epoch)
                val_results[0].append(val_loss)
                val_results[1].append(val_dice)
                # add the segmentation validation loss output here
            else:
                val_loss, val_mse, val_cosine, val_jaccard = Validate(unet, self.criterion, Val_data, epoch)
                val_results[0].append(val_loss)
                val_results[1].append(val_mse)
                val_results[2].append(val_cosine)
                val_results[3].append(val_jaccard)
                # add the regression validation loss output here
                
            print("saving epoch: ", epoch)
                        
            checkpoint = {"epoch": epoch, "state_dict": unet.state_dict(), "optimizer" : unet_opt.state_dict()}
            out = self.Total_path + "Checkpoints/" + "checkpoint_" + str(epoch) + ".pth"
            torch.save(checkpoint, out)
            
            for result_type in ["Training", "Validation"]:
                for save_location in range(len(list_of_names)):
                    
                    if not os.path.exists(self.Total_path + result_type + list_of_names[save_location][0]) == True:
                        os.makedirs(self.Total_path + result_type + list_of_names[save_location][0])
                        
                    with open(self.Total_path + result_type + list_of_names[save_location][0] + list_of_names[save_location][1], "w") as f: 
                        write = csv.writer(f) 
                        if result_type == "Training":
                            write.writerow([train_results[save_location]])
                        if result_type == "Validation":
                            write.writerow([val_results[save_location]])
                            
            if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                # loss, mse, cosine, jaccard
                train_results = [[],[],[],[]]
                val_results = [[],[],[],[]]
            else:
                # loss, dice
                train_results = [[],[]]
                val_results = [[],[]]
                
        print("Finished Training Dataset")