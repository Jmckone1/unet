class Global:
    Seed = 0
    GPU = "3"

class sData:

    image_in = "whimg_norm"
    rano_in = "RANO_2"
    index_file = "/index_max_original_2_small.npy"
    refresh_index = False

class SegNet:
    # In the format "FileName/"
    dataset_path = "Brats_2018_data/Brats_2018_data" 

    # filepath to save model details within checkpoint file
    c_file = "split_data_experiments/Full_model_MK6_H16_PTE_CO100_R3_lr_0003_dice_main_50_percent/" 

    # "Checkpoints_RANO/Unet_H16_M9_O10A0/checkpoint_99.pth" # checkpoint load path
    checkpoint_name = "Checkpoints_RANO/Unet_H16_M14_CO100_R3_main_data_input_4/checkpoint_49.pth"

    n_epochs = 3
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0003

    size = 1
    display_step = 50
    batch_size = 16
    device = 'cuda'

    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 0.5
    
    weight_decay = 1e-8
    
    extensions = ["/HGG","/LGG"]

    useWeights = True # false if with no pre-training, True with Pretraining
    allow_update = True # false if Frozen model, True if Unfrozen model
    
    checkpoint_eval = False # If True perform validation on each *display step* number of batches for the first epoch
    
    # i need to change or rerun the code without the additional sigmoid function for dice loss to see if that works.
    # i need to try the alternatives for the learning rate so that i can check that.
    # i want to but cannot check the impact of batch size at this time due to the pretrained part of the model not having the same batch size.
    # i will need to test the model outputs on the data that doesnt include tumour slices versus those that do - for comparison
    # i want to test more epochs depending on the output
    # i need to check the progressive;y smaller dataset chunks to see what impact is made
    # i need to look into the prospect of appying cross validation and methods that this can be done by
    # i need to test this on whole tumour scale rather than a per slice (random) scale that i have been doing so far
    # maabe have a play with the weight decay for the models and see what impact that has
    
class testNet:
    
    # 1:53:00 per 543 batches of slices (543*16 = 8688) 
    # (113*60) / 8688 = 0.7803 (8688 / (113*60) = 1.2814)
    # 1.2814 * 155 = 120.9599 seconds per volume. 
    # 2 mins and 1 second

#     Full_model_MK6_H16_PFE_CO100_R3_lr_0003_dice_main_01082022
#     Full_model_MK6_H16_RI_CO100_R3_lr_0003_dice_main_01082022

    # dataset_path = "Smaller_dataset/Brats_2018_data/Brats_2018_data"
    dataset_path = "Brats_2018_data/Brats_2018_data"
    extensions = ["/HGG","/LGG"]

    size = 1
    batch_size = 16
    device = 'cuda'
    
    #  Full_model_MK6_H16_RI_CO100_R3_lr_0003_dice_main_01082022
    # Full_model_MK6_H16_PFE_CO100_R3_lr_0003_dice_main_01082022
    # Full_model_MK6_H16_PTE_CO100_R3_lr_0003_dice_main_21072022
    
    load_path = "Checkpoints/split_data_experiments/1_Full_model_MK6_H16_PTE_CO100_R3_lr_0003_dice_main"
    save_path = "Predictions/MK_6_model_predictions/10_10_2022/PTE_Test_set_progression"
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0003
    weight_decay = 1e-8
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1
    
    intermediate_checkpoints = True
    end_epoch_checkpoints = True
