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
    Network = {
        "Global" : {
            "Seed" : 0,
            "device" :"cuda",
            "GPU" : 2,
            "Param_location" : "Code_UNet_2/Code_UNet_Re/Net_modules/Model_hyperparameters.py",
            "Debug" : False,
            "Net" : "UNet" # UNet | DCSAU 
            },
        "Hyperparameters" : {
            "Cosine_penalty" : 100,
            "Epochs" : 1,
            "Batch_size" : 8,
            "Learning_rate" : 3e-4,
            "Weight_decay" : 1e-8,
            "Betas" : (0.9,0.999),
            "Input_dim" : 4,
            "Label_dim" : 1,
            "Hidden_dim" : 16,
            "Train_split" : 0.7,
            "Validation_split" : 0.1,
            "Test_split" : 0.2,
            "Custom_split" : 0.5,
            "Batch_display_step" : 100,
            "Image_scale" : 1,
            "Image_size" : [240,240],
            "Evaluate" : False,
            "Regress" : False,
            "Allow_update" : True,
            "Use_weights" : False,
            "Apply_Augmentation" : False
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/Model_30_percent_input/",
            "Checkpoint_load" : "Checkpoints/UNet_reprod/Regression/checkpoint_0.pth",
            "Data_path" : "/Brats_2018_4/", # "/Brats_2018_data/Brats_2018_data",#  "/CT_Dataset/Task06_Lung", #"/Brats_2018_4/", #"/Brats_2018/", #"/Brats_2018_small/"
            "Extensions" : ["/HGG", "/LGG"]
            },
        "Test_paths" : {
            },
        "Old_Hyperparameters" : {
            "Index_File" : "/inedx_max_2.npy"
            }
        }