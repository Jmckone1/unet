# code ran on 24/05/2022 - 200 epoch code for small and large datasets
# updated code for the output of cosine, mse and combined loss seperately
###########################################################
                # Global Parameters #
###########################################################
class Global:
    Seed = 0
    GPU = "1"

###########################################################
            # RANO Dataloader Parameters #
###########################################################    
# Index filepath to save to for moving through the current non-standard sized image slices
# will create a new if the file doesnt currently exist.
class rData_Test:
    
    image_in = "whimg_norm"
    rano_in = "RANO"
    index_file = "/index_max_original.npy"
    
class rData:

    image_in = "whimg_reduced"
    rano_in = "RANO_reduced"
    index_file = "/index_max_reduced.npy"

###########################################################
         # RANO Unet Model Parameters #
###########################################################
# area penalty value is currently redundant and will not produce any impact for the penalty 2 model as it has not been implemented 
# this is purposeful until the point in time where we can test if there is any reasonable point or evidence in it working.
#
# checkpoint path in the format "FileName/" - filepath to save the network resulting checkpoint files
# size is the image interpolation multiplier
# training 70%, validation 10% and testing 20%
class rNet:
    
    checkpoint = "Unet_H16_M13_O10_cosine_loss_long/"
    dataset_path = "Brats_2018_data/Brats_2018_data"
    Extensions = ["/HGG","/LGG"]
    
    n_epochs = 200
    orth_penalty = 10
    area_penalty = 0 
    
    display_step = 200
    batch_size = 16
    lr = 0.0001
    Weight_Decay = 1e-8
    Betas = (0.9, 0.999) # not sure what this is but will look into it.
    
    input_dim = 4
    label_dim = 8
    hidden_dim = 16
    
    size = 1
    initial_shape = int(240 * size)
    target_shape = int(8)
    device = 'cuda' #this may be better in global
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1
    
class test_rNet:
    
    dataset_path = "Brats_2018_data/Brats_2018_data"
    
    display_step = True # this is responsible for confirming whether the files are saved or not
    output_path = "Unet_H16_M13_O10_cosine_loss_no_amp_mains_2"
    checkpoint_path = "Checkpoints_RANO/" + output_path + "/checkpoint_48.pth"
    Rano_save_path = "Predictions_RANO_test/newtest_maintest/"+output_path+"/RANO"
    image_save_path = "Predictions_RANO_test/newtest_maintest/"+output_path+"/IMAGE/"

###########################################################
# param file usage currently implemented in:
# RANO_dataloader_2_scandir.py (can use different naming once completed)
# UNet_RANO_Split_Mem.py
# Test_RANO_Maindata.py 
#
# Files to add
# 