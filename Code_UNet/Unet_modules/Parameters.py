# code ran on 24/05/2022 - 200 epoch code for small and large datasets
# updated code for the output of cosine, mse and combined loss seperately
###########################################################
                # Global Parameters #
###########################################################
class Global:
    Seed = 0
    GPU = "2"

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
    rano_in = "RANO_reduced_2"
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
    
    checkpoint = "Unet_H16_M13_O1000_cosine_orth_v2_small_half_v2_lr4/"
    dataset_path = "Smaller_dataset/Brats_2018_data/Brats_2018_data"
    Extensions = ["/HGG","/LGG"]
    
    n_epochs = 100
    orth_penalty = 1000
    area_penalty = 0 
    
    display_step = 100
    batch_size = 16
    lr = 0.0004
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
    output_path = "Unet_H16_M13_O10_cosine_orth_v2_main_half_v2"
    checkpoint_path = "Checkpoints_RANO/" + output_path + "/checkpoint_47.pth"
    Rano_save_path = "Predictions_RANO_test/newtest_maintest/"+output_path+"/RANO/"
    image_save_path = "Predictions_RANO_test/newtest_maintest/"+output_path+"/IMAGE/"

###########################################################
# param file usage currently implemented in:
# RANO_dataloader_2_scandir.py (can use different naming once completed)
# UNet_RANO_Split_Mem.py
# Test_RANO_Maindata.py 
#
# Files to add
# 



# what do i need to be doing next - 06/06/22
# i need to check the penalty for cosine on more examples to see what its doing
# i need to run the model cosine at a much higher weighting to see what the difference is
# - ran the smaller dataset at 100 and 1000 cosine multilier to see the impact of this. in theory we should see it minimising more and the mse minimising less in extreme cases.

# hopefully this will work or at least show something in the next day or so

# im going to pick one of the example checkpoints that i have so far and run the segmentation code to test that the update fucntions correctly.

# i want to have a look at reducing the dimesions of the input data to just the flair channel and seeing what impact that makes - can also look at each of the pother channel variations - will have to make a second version of all the code bases to test this - can from there also work on the 2 channels and comboniations and then 3 channel combinations - will have to write down all permutations

# increasing the value of the cosine weighting provides very little improvement if any change at all.

# there is currently a Failed to initialize NVML: Unknown Error error that is preventing my docker cotainer from detecting the GPU so i cannot run any further models until i restrt it at the very least. not sure why this has occured.

# the next thing i want to try is to increase the learning rate - since the model takes a long time to converge i am hoping this will help.

# again, check that the segemntation is functional - even in a terrible manner - this section does not need to be fully complete at this stage i just need to see if it is functional in general and that my therories still do what i think they should be doing.