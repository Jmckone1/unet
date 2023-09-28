#       # -*- coding: utf-8 -*-
# """
# @author: Joshua Mckone
# """

# class Parameters():
#     Network = {
#         "Global" : {
#             "Seed" : 11,
#             "device" :"cuda",
#             "GPU" : 3,
#             "Param_location" : "Code_UNet_2/Code_UNet_Re/Net_modules/Model_hyperparameters.py",
#             "Debug" : False,
#             "Net" : "UNet",
#             "Enable_Determinism" : True
#             },
#         "Hyperparameters" : {
#             "Cosine_penalty" : 2,
#             "Epochs" : 10,
#             "Batch_size" : 8,
#             "Learning_rate" : 3e-4,
#             "Weight_decay" : 1e-8,
#             "Betas" : (0.9,0.999),
#             "Input_dim" : 4,
#             "Label_dim" : 1,
#             "Hidden_dim" : 16,            
#             #12.5,25,37.5,50,62.5,75,87.5,100
#             "Train_split" : [0,1,2,3,4,5,6],
#             "Validation_split" : [7],
#             "Test_split" : [8,9],
#             "Custom_split" : [0,1,2,3,4,5],
#             ############################################
#             "Batch_display_step" : 100,
#             "Image_scale" : 1,
#             "Image_size" : [240,240],
#             "Evaluate" : False,
#             "Regress" : False,
#             "BBox": False,
#             "RANO": True,
#             "Allow_update" : True,
#             "Use_weights" : True,
#             "Apply_Augmentation" : False,
#             "Single_channel" : False,
#             "Single_channel_type" : "Flair", # T1, Flair, T1ce, T2
#             "Load_Checkpoint" : False
#             },
#         "Train_paths" : {
#             "Checkpoint_save" : "Checkpoints/RANO_cosine_cont_100_PTE/",
#             "Checkpoint_load" : "Checkpoints_RANO/Checkpoints/cosine_absolute_values_c_100_e100_Continue/checkpoint_47.pth",
#             "Data_path" : "/Datasets/Brats_2018_4/",
#             "Extensions" : ["/HGG","/LGG"] # ["/HGG", "/LGG"]
#             },
#         "Test_paths" : {
#             "Intermediate_checkpoints" : False,
#             "End_epoch_checkpoints" : True,
#             "Data_path" : "/Datasets/Brats_2018_4/", # "/Datasets/CT_Dataset/Task06_Lung",#
#             "Extensions" : ["/HGG","/LGG"], #["/HGG","/LGG"],
#             "Epochs" : 10,
#             "Load_path" : "Checkpoints/Checkpoints/Brats_5_c_100_pretrained_10_epochs/",
#             #CT_randomInit_10_c_100_Prano_10_epochs CT_pretrain_10_c_100_Prano_10_epochs
#             "Save_path" : "Test_outputs/test/" # ct data, random init, 10% data, cosine 100, 10 epochs
#             },
#         "Old_Hyperparameters" : {
#             "Index_File" : "/inedx_max_2.npy"
#             }
#         }

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
            "Cosine_penalty" : 10,
            "Epochs" : 50,
            "Batch_size" : 8,
            "Learning_rate" : 3e-4,
            "Weight_decay" : 1e-8,
            "Betas" : (0.9,0.999),
            "Input_dim" : 4,
            "Label_dim" : 1,
            "Hidden_dim" : 16,            
            "Train_split" : [0,1,2,3,4,5,6],
            "Validation_split" : [7],
            "Test_split" : [8,9],
            "Custom_split" : [0,1,2,3,4,5],
            "Batch_display_step" : 100,
            "Image_scale" : 1,
            "Image_size" : [240,240],
            "Evaluate" : True,
            "Regress" : False,
            "BBox": False,
            "RANO": True,
            "Allow_update" : True,
            "Use_weights" : False,
            "Apply_Augmentation" : False,
            "Single_channel" : False,
            "Single_channel_type" : "Flair",
            "Load_Checkpoint" : True
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/cosine_absolute_values_c_10_e100_Continue/",
            "Checkpoint_load" : "Checkpoints/cosine_absolute_values_c0_e100/checkpoint_41.pth",
            "Data_path" : "/Datasets/Brats_2018_4/",
            "Extensions" : ["/HGG","/LGG"]
            },
        "Test_paths" : {
            "Intermediate_checkpoints" : False,
            "End_epoch_checkpoints" : True,
            "Data_path" : "/Datasets/Brats_2018_4/",
            "Extensions" : ["/HGG","/LGG"],
            "Epochs" : 10,
            "Load_path" : "Checkpoints/Checkpoints/RANO_cosine_cont_3_PTE/",
            "Save_path" : "Test_outputs/PTE_cosine_3_100data/"
        },
        "Old_Hyperparameters" : {
            "Index_File" : "/inedx_max_2.npy"
            }
        }