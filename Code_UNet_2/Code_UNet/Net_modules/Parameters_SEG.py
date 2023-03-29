# -*- coding: utf-8 -*-
"""
@author: Joshua Mckone
"""

# consider changing the outline of the parameters and therefore the other code composites to split the hyperparameters file

# check the hyperparamters used in the regression versus the segmentation and note them down
# check whether the parameter file is saved as a copy so that these changes can be tracked

# Segmentation and regression show similar parameters being the best for each variation - at least on the 
# lr = 0.0003
# weight_decay = 1e-8

class Parameters():
    PRANO_Net = {
        "Global" : {
            "Seed" : 0,
            "device" :"cuda",
            "GPU" : 2,
            "Param_location" : "Code_UNet_2/Code_UNet/Net_modules/Parameters_SEG.py",
            "Debug" : False,
            "Net" : "UNet"
            },
        "Hyperparameters" : {
            "Cosine_penalty" : 100,
            "Epochs" : 100,
            "Batch_size" : 8,
            "Learning_rate" : 3e-4,
            "Weight_decay" : 1e-8,
            "Betas" : (0.9,0.999),
            "Input_dim" : 1,
            "Label_dim" : 1,
            "Hidden_dim" : 16,
            "Train_split" : 0.7,
            "Validation_split" : 0.1,
            "Test_split" : 0.2,
            "Custom_split" : 0.5,
            "Batch_display_step" : 100, # used in line 228 of Unet_full_train file
            # "Confirm_display_step" : True, # not sure if this is used - will need to check
            "Image_scale" : 1,
            "Image_size" : [240,240],
            "Evaluate" : False,
            "Regress" : True,
            "Allow_update" : True,
            "Use_weights" : False,
            "Apply_Augmentation" : True
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/UNet_brats",
            "Checkpoint_load" : "Checkpoints/UNet_brats/Regression/checkpoint_0.pth",
            "Data_path" : "/Brats_2018/", # "CT_Dataset/Task06_Lung",
            "Extensions" : [""]
            },
        "Test_paths" : { # test isnt yet implemented so none of these are used
#             "Checkpoint_load" : "Checkpoints/UNet_H16_M15_C100_LR4/checkpoint_199",
#             "Data_path" : "Task06_Lung",
#             "Predictions_output" : "Predictions/UNet_H16_M15_C100_LR4",
#             "PRANO_save" : "/PRANO",
#             "Image_save" : "/IMAGE"
            }
        }
