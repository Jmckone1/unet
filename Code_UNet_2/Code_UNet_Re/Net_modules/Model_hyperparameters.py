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
            "Epochs" : 100,
            "Batch_size" : 8,
            "Learning_rate" : 3e-4,
            "Weight_decay" : 1e-8,
            "Betas" : (0.9,0.999),
            "Input_dim" : 4,
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
            "Image_size" : [240,240],
            "Evaluate" : False,
            "Regress" : True,
            "BBox": True,
            "RANO": False,
            "Allow_update" : True,
            "Use_weights" : False,
            "Apply_Augmentation" : False,
            "Single_channel" : False,
            "Single_channel_type" : "Flair" # T1, Flair, T1ce, T2
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/BBox_100_epochs_100_brats_4_v0/",
            "Checkpoint_load" : "Checkpoints_RANO/Checkpoints/Prano_pretrain_1_C100/checkpoint_40.pth",
            "Data_path" : "/Datasets/Brats_2018_4/",#"/Datasets/CT_Dataset/Task06_Lung", # "/Brats_2018_data/Brats_2018_data",#  "/CT_Dataset/Task06_Lung", #"/Brats_2018_4/", #"/Brats_2018/", #"/Brats_2018_small/"
            "Extensions" : ["/HGG","/LGG"] # ["/HGG", "/LGG"]
            },
        "Test_paths" : {
            "Intermediate_checkpoints" : False,
            "End_epoch_checkpoints" : True,
            "Data_path" : "/Datasets/Brats_2018_4/", # "/Datasets/CT_Dataset/Task06_Lung",#
            "Extensions" : ["/HGG","/LGG"], #["/HGG","/LGG"],
            "Epochs" : 10,
            "Load_path" : "Checkpoints/Checkpoints/Brats_100_c_100_pretrained_10_epochs/",
            #CT_randomInit_10_c_100_Prano_10_epochs CT_pretrain_10_c_100_Prano_10_epochs
            "Save_path" : "Test_outputs/Brats_PTE_D100_C100_E10_02/" # ct data, random init, 10% data, cosine 100, 10 epochs
            },
        "Old_Hyperparameters" : {
            "Index_File" : "/inedx_max_2.npy"
            }
        }