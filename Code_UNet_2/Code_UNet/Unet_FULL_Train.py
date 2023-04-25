from Net_modules.Evaluation import Dice_Evaluation as Dice_Eval
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
import Net_modules.Parameters_SEG as Param

if Param.Parameters.PRANO_Net["Global"]["Net"] == "DCSAU":
    import Net.pytorch_dcsaunet.DCSAU_Net as net
elif Param.Parameters.PRANO_Net["Global"]["Net"] == "UNet":
    import Net.UNet_components as net
else:
    import sys
    print("Network *" + Param.Parameters.PRANO_Net["Global"]["Net"] + "* not implemented - please try one of the following:")
    print("* DCSAU *")
    print("* UNet *")
    sys.exit()
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
        
        self.model_improvement = 0
        
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
        Improvement = 0
        
        sigmoid_act = nn.Sigmoid()
        
        # regress swaps the model between regression and the segmentation model
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Use_weights"] == True:
            unet = net.Model.load_weights(Param.Parameters.PRANO_Net["Hyperparameters"]["Input_dim"],
                                          Param.Parameters.PRANO_Net["Hyperparameters"]["Label_dim"],
                                          Regress = Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"],
                            Allow_update = Param.Parameters.PRANO_Net["Hyperparameters"]["Allow_update"],
                            Checkpoint_name = Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_load"]).to(
                Param.Parameters.PRANO_Net["Global"]["device"])
        else:
            unet = net.Model(Param.Parameters.PRANO_Net["Hyperparameters"]["Input_dim"],
                             Param.Parameters.PRANO_Net["Hyperparameters"]["Label_dim"],
                             Regress = Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"]).to(
                Param.Parameters.PRANO_Net["Global"]["device"])
            
        if self.Debug: print("Created model", torch.cuda.memory_allocated()/1024**2)
        
        print(unet)
        with open( self.Total_path + "/" + "Model_architecture", "w") as write: 
            write.write(str(unet))
            
        unet_opt = torch.optim.Adam(unet.parameters(), lr=Param.Parameters.PRANO_Net["Hyperparameters"]["Learning_rate"], betas=Param.Parameters.PRANO_Net["Hyperparameters"]["Betas"],weight_decay=Param.Parameters.PRANO_Net["Hyperparameters"]["Weight_decay"])
        
        scaler = amp.GradScaler(enabled = True)
    
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
                                 ["_Jaccard/","epoch_" + str(epoch) + "_jaccard_index.csv"],
                                 ["_loss_cosine/","epoch_" + str(epoch) + "_loss_mse.csv"],
                                 ["_loss_mse/","epoch_" + str(epoch) + "_loss_cosine.csv"]]
            else:
                # loss, dice
                train_results = [[],[]]
                val_results = [[],[]]
                
                list_of_names = [["_loss/","epoch_" + str(epoch) + "_loss.csv"],
                             ["_Dice/","epoch_" + str(epoch) + "_dice_score.csv"]]
            
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
                
                truth_input = truth_input.to(dtype=torch.half)
                label_input = label_input.to(dtype=torch.half)
                if self.Debug: print("Input label data", torch.cuda.memory_allocated()/1024**2)
                
                if(truth_input.ndim == 3):
                    truth_input = truth_input[:,np.newaxis,:,:]
                    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
                        label_input = label_input[:,np.newaxis,:,:]
                    if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                        label_input = label_input[:,np.newaxis,:]
                
                # set accumilated gradients to 0 for param update
                unet_opt.zero_grad()# set_to_none=True)
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
#                     limits = torch.finfo(torch.float16)
#                     print(limits.min,limits.max)
#                     unet_loss = torch.clamp(self.criterion(pred, label_input), min=torch.float(limits.min), max=torch.float(limits.max))
                    unet_loss = self.criterion(pred, label_input)
                    
                truth_output = label_input.cpu().detach().numpy().squeeze()
                
                if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:
                    # calculate jaccard score
                    pred_output = pred.cpu().detach().numpy().squeeze()
                    
                    
                        
                    if self.Debug: print("Pred Magenta, Truth Yellow")
                    for Batch in range(cur_batch_size):
                        if self.Debug:
                            print("input", truth_output[Batch,:])
                            print("prediction", pred_output[Batch,:])
                        corners_truth, center_truth = Jacc.Obb(truth_output[Batch,:])
                        mask_truth = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"], corners_truth)
                        corners_pred, center_pred = Jacc.Obb(pred_output[Batch,:])
                        mask_pred = Jacc.mask(Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],   corners_pred)
#                         print("Pred then truth",pred_output,truth_output)
#                         print("")
                        if np.sum(np.sum(mask_pred)) > 2:
                            train_results[1].append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                        else:
                            train_results[1].append(float(0)) # "NaN")) # should this be nan or 0

                        if self.Debug: print("Jaccard score: ", train_results[1][-1]) 
                        
                    if self.Debug:
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
                    
#                         print("Truth_output")
                            
                    unet_cosine = unet_loss[2]
                    unet_mse = unet_loss[1]
                    # reset unet loss to a non array structure
                    unet_loss = unet_loss[0]
                    if self.Debug:
                        print("training loss", unet_loss)
                        print("training jaccard", train_results[1][-1])
                        print("training cosine", unet_cosine)
                        print("training mse", unet_mse)
                else:
                    #calculate dice score
                    
                    pred_output = pred.cpu().detach().numpy()
                    for Batch in range(cur_batch_size):
                        if self.Debug: print("DICE SCORE: ", Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:]))

                        train_results[1].append(Dice_Eval.dice_score((pred_output[Batch,:,:] > 0.5).astype(int),truth_output[Batch,:,:]))
                        if self.Debug:
                            fig = plt.figure(figsize=(10,6))
                            grid = ImageGrid(fig, 111,nrows_ncols=(2, 4),axes_pad=0.1)

                            for ax, im in zip(grid, truth_output):
                                print(np.min(im),np.max(im))
                                ax.imshow(im,cmap='gray')#, vmin=0, vmax=1)
                            print("Truth_output")
                            plt.show()   

                            fig2 = plt.figure(figsize=(10,6))
                            grid2 = ImageGrid(fig2,111,nrows_ncols=(2, 4),axes_pad=0.1)

                            for ax2, im2 in zip(grid2, (pred_output > 0.5).astype(int)):
                                print(np.min(im2),np.max(im2))
                                ax2.imshow(im2,cmap='gray')#, vmin=0, vmax=1)
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
                    
                    train_results[2].append(running_cosine)
                    train_results[3].append(running_mse)
                            
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
                val_results[1].append(val_jaccard)
                val_results[2].append(val_cosine)
                val_results[3].append(val_mse)
                # add the regression validation loss output here
                
            print("saving epoch: ", epoch)
            
            # if the Dice score or Jaccard score for the corresponsding model during training is higher then save checkpoint
#             print("List of values", train_results[0])
#             print("first value", train_results[0])
#             print("Last 8 values", train_results[0][-8], train_results[1][-8])
            print("Previous score: ", self.model_improvement, "New score: " , train_results[0][-1])
            if epoch == 0:
                self.model_improvement = train_results[0][-1]
            if train_results[0][-1] <= self.model_improvement:
                print("Updating improvement value")

                checkpoint = {"epoch": epoch, "state_dict": unet.state_dict(), "optimizer" : unet_opt.state_dict()}
                out = self.Total_path + "Checkpoints/" + "checkpoint_" + str(epoch) + ".pth"
                torch.save(checkpoint, out)

                self.model_improvement = train_results[0][-1]
            else:
                print("Not updating improvement value")
            
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