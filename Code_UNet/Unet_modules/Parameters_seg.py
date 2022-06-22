class Global:
    Seed = 0
    GPU = "1"
    
class sData:

    image_in = "whimg_norm"
    rano_in = "RANO_2"
    index_file = "/index_max_original.npy"
    
class SegNet:
    
    dataset_path = "Brats_2018_data/Brats_2018_data" # In the format "FileName/"
    c_file = "split_data_experiments/Full_model_MK6_H16_PFE_CO100_R3_v2/" # filepath to save model details within checkpoint file
    checkpoint_name = "Checkpoints_RANO/Unet_H16_M14_CO100_R3_main_data_input_4/checkpoint_49.pth"
    # "Checkpoints_RANO/Unet_H16_M9_O10A0/checkpoint_99.pth" # checkpoint load path

    n_epochs = 10
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0002
    
    size = 1
    display_step = 50
    batch_size = 16
    device = 'cuda'
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1
    
    weight_decay = 1e-8
    
    extensions = ["/HGG","/LGG"]

    useWeights = True # false if with no pre-training, True with Pretraining
    allow_update = False # false if Frozen model, True if Unfrozen model