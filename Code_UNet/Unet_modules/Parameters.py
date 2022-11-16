# code ran on 24/05/2022 - 200 epoch code for small and large datasets
# updated code for the output of cosine, mse and combined loss seperately
###########################################################
                # Global Parameters #
###########################################################

class Global:
    Seed = 0

###########################################################
            # RANO Dataloader Parameters #
###########################################################
class rData:


    image_in = "whimg_reduced"
    rano_in = "RANO_reduced_2"
    index_file = "/index_max_reduced.npy"

###########################################################
         # RANO Unet Model Parameters #
###########################################################
class rNet:
    
    checkpoint = "Unet_H16_M14_CO0_R3_main_data_0_000003/"
    dataset_path = "Brats_2018_data/Brats_2018_data"
    Extensions = ["/HGG","/LGG"]
    
    n_epochs = 50
    orth_penalty = 0
    area_penalty = 0 
    
    display_step = 100
    batch_size = 16
    lr = 0.000003
    Weight_Decay = 1e-8
    Betas = (0.9, 0.999) # not sure what this is but will look into it.
    
    input_dim = 4
    label_dim = 8
    hidden_dim = 16
    
    size = 1
    initial_shape = int(240 * size)
    target_shape = int(8)
    device = 'cuda'

    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1
    
class test_rNet:
    
    dataset_path = "Smaller_dataset/Brats_2018_data/Brats_2018_data"
    
    display_step = True # this is responsible for confirming whether the files are saved or not
    output_path = "Unet_H16_M14_CO100_R3_main_data_input_4"
    checkpoint_path = "Checkpoints_RANO/" + output_path + "/checkpoint_49.pth"
    Rano_save_path = "Predictions_RANO_test/newtest_maintest/" + output_path + "2/RANO/"
    image_save_path = "Predictions_RANO_test/newtest_maintest/" + output_path + "2/IMAGE/"

###########################################################
# param file usage currently implemented in:
# RANO_dataloader_2_scandir.py (can use different naming once completed)
# UNet_RANO_Split_Mem.py

# Files to add
# Test_RANO_2.py

# # # ##########################################################
# # # param file usage currently implemented in:
# # # RANO_dataloader_2_scandir.py (can use different naming once completed)
# # # UNet_RANO_Split_Mem.py
# # # Test_RANO_Maindata.py 

# # # Files to add

# # # what do i need to be doing next - 06/06/22
# # # i need to check the penalty for cosine on more examples to see what its doing
# # # i need to run the model cosine at a much higher weighting to see what the difference is
# # # - ran the smaller dataset at 100 and 1000 cosine multilier to see the impact of this. in theory we should see it minimising more and the mse minimising less in extreme cases.

# # # hopefully this will work or at least show something in the next day or so

# # # im going to pick one of the example checkpoints that i have so far and run the segmentation code to test that the update fucntions correctly.

# # # i want to have a look at reducing the dimesions of the input data to just the flair channel and seeing what impact that makes - can also look at each of the pother channel variations - will have to make a second version of all the code bases to test this - can from there also work on the 2 channels and comboniations and then 3 channel combinations - will have to write down all permutations

# # # increasing the value of the cosine weighting provides very little improvement if any change at all.

# # # there is currently a Failed to initialize NVML: Unknown Error error that is preventing my docker cotainer from detecting the GPU so i cannot run any further models until i restrt it at the very least. not sure why this has occured.

# # # the next thing i want to try is to increase the learning rate - since the model takes a long time to converge i am hoping this will help.

# # # again, check that the segemntation is functional - even in a terrible manner - this section does not need to be fully complete at this stage i just need to see if it is functional in general and that my therories still do what i think they should be doing.

# # All of the above has been completed as of 15/06/22

# # 22/06/22
# # for some reason the predicted NPZ files for RANO measurements are not appearing and whilst the folder remains visibly empty the file cannot be deleted because it contains something, possible error with .filename? will need to check ls to find out.
# # also there is no prediction of the LGG examples in the files which is also an issue that needs to be checked to see if it is a problem or not, ergo finding out if the randomised fiel contain both LGG and HGG or just HGG (which is unlikely but possible)

# # there also is a potential memory leak within the segmentaion code, its only minor at this point and may be related to the files that track the output that is subsequently saved to the output - im thinking that i will append the vlaues straight into the save folder and then remove reference to it from the running code instead of keeping one large array that is appended to as we run, this makes the code take longer and longer to run as we go on. i will solve this and see if the problem is still in place. if it is then i will look at both the optimisation code that andy suggested and will see what else i can find in the code that may be causing this issue.

# # need to make sure to have a look at and ask about conferences before and during the meeting on friday. both in terms of submitting the work that i am currently doing and for purposes of attending. i feel like ive lost track alot of what ive been doing and lost touch in where to be going next, instead of being stuck down the rabbit hoel with my current work tog et it out there. i did attend the research showcase which is good but could be better.

# people having been asking how loing it takes to run the model and what resources. so this is something that im going to need to add to the logging file i think, start-runtime and end-runtime. plus make sure to minus one from the other and take validation into account.
>>>>>>> Param
