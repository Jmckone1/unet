      # -*- coding: utf-8 -*-
"""
@author: Joshua Mckone
"""

class Parameters():
    Network = {
        "Global" : {
            "Seed" : 11,
            "device" :"cuda",
            "GPU" : 2,
            "Param_location" : "Code_UNet_2/Code_UNet_Re/Net_modules/Model_hyperparameters.py",
            "Debug" : False,
            "Net" : "UNet",
            "Enable_Determinism" : True
            },
        "Hyperparameters" : {
            "Cosine_penalty" : 100,
            "Epochs" : 10,
            "Batch_size" : 8,
            "Learning_rate" : 3e-4,
            "Weight_decay" : 1e-8,
            "Betas" : (0.9,0.999),
            "Input_dim" : 1,
            "Label_dim" : 1,
            "Hidden_dim" : 16,            
            #12.5,25,37.5,50,62.5,75,87.5,100
            "Train_split" : [0,1,2,3,4,5,6],
            "Validation_split" : [7],
            "Test_split" : [8,9],
            "Custom_split" : [0,1,2,3,4,5],
            ############################################
            "Batch_display_step" : 100,
            "Image_scale" : 1,
            "Image_size" : [256,256],
            "Evaluate" : False,
            "Regress" : False,
            "Allow_update" : True,
            "Use_weights" : False,
            "Apply_Augmentation" : False
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/CT_pretrain_100_c_100_Prano_10_epochs/",
            "Checkpoint_load" : "Checkpoints_RANO/Checkpoints/CT_50_c_100_Prano_50_epochs/checkpoint_33.pth",
            "Data_path" : "/Datasets/CT_Dataset/Task06_Lung", # "/Brats_2018_data/Brats_2018_data",#  "/CT_Dataset/Task06_Lung", #"/Brats_2018_4/", #"/Brats_2018/", #"/Brats_2018_small/"
            "Extensions" : [""] # ["/HGG", "/LGG"]
            },
        "Test_paths" : {
            "Intermediate_checkpoints" : False,
            "End_epoch_checkpoints" : True,
            "Data_path" :  "/Datasets/CT_Dataset/Task06_Lung",#"/Brats_2018_4/",
            "Extensions" : [""], #["/HGG","/LGG"],
            "Epochs" : 5,
            "Load_path" : "Checkpoints/Checkpoints/CT_randomInit_10_c_100_Prano_10_epochs/",
            "Save_path" : "Test_outputs/CT_RI_D10_C100_E10/" # ct data, random init, 10% data, cosine 100, 10 epochs
            },
        "Old_Hyperparameters" : {
            "Index_File" : "/inedx_max_2.npy"
            }
        }