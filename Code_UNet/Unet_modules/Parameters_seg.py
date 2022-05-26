class Global:
    Seed = 0
    GPU = "1"
    
class SegNet:
    
    # In the format "FileName/"
    c_file = "split_data_experiments/Full_model_MK5_H16_baseline_6_epochs_BCELoss_2/"
    checkpoint_name = "Checkpoints_RANO/Unet_H16_M9_O10A0/checkpoint_99.pth"

    n_epochs = 6
    input_dim = 4
    label_dim = 1
    hidden_dim = 16
    lr = 0.0002
    
    size = 1
    display_step = 50
    batch_size = 16
    initial_shape = int(240 * size)
    target_shape = int(240 * size)
    device = 'cuda'
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    
    weight_decay = 1e-8
    
    