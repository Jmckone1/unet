###########################################################
            # RANO Dataloader Parameters #
###########################################################
class rData:

    # Input filepaths for the dataloader
    image_in = "whimg_reduced"
    rano_in = "RANO_reduced"
    # Index filepath to save to for moving through the current non-standard sized image slices - will create a new if the file doesnt currently exist.
    index_file = "/index_max_reduced.npy"

###########################################################
     # RANO Unet Model Parameters #
###########################################################

class rNet:

    # In the format "FileName/" - filepath to save the network resulting checkpoint files as
    c_file = "Unet_H16_M13_O10/"
    dataset_path = "Brats_2018_data/Brats_2018_data"

    # image interpolation multiplier
    # this does not work at this time for the RANO implementation
    size = 1

    # inital testing showed 50 as the best estimated region before plautau though this may change.
    n_epochs = 100
    input_dim = 4
    label_dim = 8
    hidden_dim = 16
    orth_penalty = 10
    area_penalty = 0 
    # area penalty value is currently redundant and will not produce any impact for the penalty 2 model as it has not been implemented - this is purposeful until the point in time where we can test if there is any reasonable point or evidence in it working.

    display_step = 200
    batch_size = 16
    lr = 0.0001
    initial_shape = int(240 * size)
    target_shape = int(8)
    device = 'cuda'
    
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    custom_split_amount = 1

###########################################################
# param file usage currently implemented in:
# RANO_dataloader_2_scandir.py (can use different naming once completed)