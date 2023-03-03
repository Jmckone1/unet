class Global:
    Seed = 0
    GPU = "0"

class rData:

    image_in = "whimg_reduced"
    rano_in = "RANO_reduced_2"
    index_file = "/index_max_reduced.npy"

class rNet:
    
    checkpoint = "Unet_H16_M14_CO0_R3_main_data_0_000003/"
    dataset_path = "Brats_2018_data/Brats_2018_data"
    Extensions = ["/HGG","/LGG"]
    
    n_epochs = 200
    orth_penalty = 0
    area_penalty = 0 
    
    display_step = 100
    batch_size = 16
    lr = 0.000003
    Weight_Decay = 1e-8
    Betas = (0.9, 0.999)
    
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