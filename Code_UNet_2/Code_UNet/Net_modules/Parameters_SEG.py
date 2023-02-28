# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:31:02 2022

@author: Computing
"""

class Parameters():
    PRANO_Net = {
        "Global" : {
            "Seed" : 0,
            "device" :"cuda",
            "GPU" : 2
            },
        "Hyperparameters" : {
            "Cosine_penalty" : 100,
            "Epochs" : 200,
            "Batch_size" : 32,
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
            "Batch_display_step" : 100,
            "Confirm_display_step" : True,
            "Image_scale" : 1,
            "New_index" : True,
            "Image_size" : [240,240],
            "Evaluate" : False,
            "Regress" : False
            },
        "Train_paths" : {
            "Checkpoint_save" : "Checkpoints/UNet_DCSAU_Net_version_small_test",
            "Data_path" : "/Data/",
            "Extensions" : ["LGG/","HGG/"],
            "Index_file" : "\\experiment_1\\index_small.npy"
            },
        "Test_paths" : {
            "Checkpoint_load" : "Checkpoints/UNet_H16_M15_C100_LR4/checkpoint_199",
            "Data_path" : "Task06_Lung",
            "Predictions_output" : "Predictions/UNet_H16_M15_C100_LR4",
            "PRANO_save" : "/PRANO",
            "Image_save" : "/IMAGE"
            }
        }
