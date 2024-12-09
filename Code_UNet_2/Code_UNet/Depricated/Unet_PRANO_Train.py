import Net.UNet_PRANO_components as net
import Net_modules.Parameters_PRANO as Param
from Net_modules.Evaluation import Jaccard_Evaluation as Jacc

from Unet_PRANO_Validate import Validate

from sklearn.metrics import jaccard_score
import torch.cuda.amp as amp
import numpy as np
import shutil
import torch
from tqdm import tqdm
import os
import csv

class UNet_train(): 
    def __init__(self, criterion):
        print("Init")
        
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
        
        criterion = self.criterion
        Improvement = 0
        scaler = amp.GradScaler(enabled = True)
        
        unet = net.UNet(Param.Parameters.PRANO_Net["Hyperparameters"]["Input_dim"], 
                        Param.Parameters.PRANO_Net["Hyperparameters"]["Label_dim"], 
                        Param.Parameters.PRANO_Net["Hyperparameters"]["Hidden_dim"]).to(
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
            mse_run = 0.0
            cosine_run = 0.0
            
            # loss, mse, cosine, jaccard
            train_results = [[],[],[],[]]
            val_results = [[],[],[],[]]

            for truth_input, label_input in tqdm(Train_datas):
    
                cur_batch_size = len(truth_input)
                
                # flatten ground truth and label masks
                truth_input = truth_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
                truth_input = truth_input.float() 
                truth_input = truth_input.squeeze()
                label_input = label_input.to(Param.Parameters.PRANO_Net["Global"]["device"])
                label_input = label_input.float()
                label_input = label_input.squeeze()
    
                # set accumilated gradients to 0 for param update
                unet_opt.zero_grad()
                pred = unet(truth_input)
                pred = pred.squeeze()
    
                # forward
                unet_loss, mse, cosine = criterion(pred, label_input)
                
                # calculate jaccard score
                for input_val in range(cur_batch_size):
                    
                    corners_truth, center_truth = Jacc.Obb(label_input[input_val,:])
                    mask_truth = Jacc.mask((240,240),corners_truth)*1
                    corners_pred, center_pred = Jacc.Obb(pred[input_val,:])
                    mask_pred = Jacc.mask((240,240),corners_pred)*1
                    
                    if np.sum(np.sum(mask_pred)) > 2:
                        train_results[3].append(jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary'))
                    else:
                        train_results[3].append(float("NaN"))
                
                # backward
                scaler.scale(unet_loss).backward()
                scaler.step(unet_opt)
                scaler.update()

                running_loss =+ unet_loss.item()
                
                mse_run =+ mse.item()
                cosine_run =+ cosine.item() 
                
                train_results[0].append(running_loss)
                train_results[1].append(mse_run)
                train_results[2].append(cosine_run)
                cur_step += 1
                
            epoch_val_loss, epoch_valid_mse, epoch_valid_cosine, epoch_jaccard_valid = Validate(unet, criterion, Val_data)
            val_results[0].append(epoch_val_loss)
            val_results[1].append(epoch_valid_mse)
            val_results[2].append(epoch_valid_cosine)
            val_results[3].append(epoch_jaccard_valid)
            
            print(epoch_val_loss)
            print(epoch_valid_mse)
            print(epoch_valid_cosine)
            
            print("Improvement: ", Improvement)
            print("Nan mean jaccard validation over the epoch: ", np.nanmean(epoch_jaccard_valid))
            print("Mean jaccard over epoch with nan: ", epoch_jaccard_valid)
            print("")
            
            # save a checkpoint only if there has been an improvement in the total jaccard score for the model.
            if np.nanmean(epoch_jaccard_valid) > Improvement:
                if np.nanmean(epoch_jaccard_valid) == np.isnan:
                    Improvement = 0
                else:
                    Improvement = np.nanmean(epoch_jaccard_valid)
            
                print("saving epoch: ", epoch)
                checkpoint = {'epoch': epoch, 'state_dict': unet.state_dict(), 'optimizer' : unet_opt.state_dict()}
                out = "Checkpoints_RANO/" + Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + "checkpoint_" + str(epoch) + ".pth"
                torch.save(checkpoint, out)

            list_of_names = ["_loss/epoch_" + str(epoch) + "loss.csv",
                             "_loss_mse/epoch_" + str(epoch) + "loss_mse.csv",
                             "_loss_cosine/epoch_" + str(epoch) + "loss_cosine.csv",
                             "_Jaccard/epoch_" + str(epoch) + "jaccard_index.csv"]
            
            for result_type in ["Training", "Validation"]:
                for save_location in range(len(list_of_names)):
                    with open("Checkpoints_RANO/" + 
                              Param.Parameters.PRANO_Net["Train_paths"]["Checkpoint_save"] + 
                              result_type + 
                              list_of_names[save_location], 
                              'w') as f: 
                        
                        write = csv.writer(f) 
                        if result_type == "Training":
                            write.writerow(train_results[save_location])
                        if result_type == "Validation":
                            write.writerow(val_results[save_location])
                            
            train_results = [[],[],[],[]]
            val_results = [[],[],[],[]]
    
        print('Finished Training Dataset')