import DeepNeuro_normalise as Deep_norm

def norm_one_channel():
    path = "Brats_2018_data/Brats_2018_data"
    path_ext = ["/HGG", "/LGG"]
    data_out = ["t2_norm",]
    filetype = ["t2"]

    # Normalization.RunDataset(path, path_ext, data_out, filetype, save = True, Log_output = True, remove = True)
    Deep_norm.Normalization.Single_norm(path, path_ext, data_out, filetype, save = True, Log_output = False)

def reduce_one_channel():
    x = 1
    
def main():
    norm_one_channel()