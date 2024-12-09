class Global:
    Seed = 0
    GPU = "3"

class sData:

    image_in = "whimg_norm"
    rano_in = "RANO_2"
    index_file = "/inedx_max_2.npy"
    refresh_index = False

class SegNet:
    
    dataset_path = "Brats_2018_data/Brats_2018_data" 
    c_file = "Checkpoints/Brats_Test_1_minimised_model_ODL/"
    checkpoint_name = "Checkpoints_RANO/Unet_H16_M14_CO100_R3_main_data_input_4/checkpoint_49.pth"

    n_epochs = 1
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0003

    size = 1
    display_step = 50
    batch_size = 32
    device = 'cuda'

    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1
    
    weight_decay = 1e-8
    
    extensions = ["/HGG","/LGG"]

    useWeights = False # false if with no pre-training, True with Pretraining
    allow_update = True # false if Frozen model, True if Unfrozen model
    
    checkpoint_eval = False # If True perform validation on each *display step* number of batches for the first epoch

class testNet:

    dataset_path = "Brats_2018_data/Brats_2018_data"
    extensions = ["/HGG","/LGG"]

    size = 1
    batch_size = 16
    device = 'cuda'

    load_path = "Checkpoints/split_data_experiments/RI_division_AS_BN_S_1/Full_model_MK6_H16_RI_CO100_R3_lr_0003_dice_main_100_percent"
    save_path = "Predictions/MK_7_model_predictions/RI_division_AS_BN_S_1/RI_100%"
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0003
    weight_decay = 1e-8
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 0.1
    
    intermediate_checkpoints = False
    end_epoch_checkpoints = True
