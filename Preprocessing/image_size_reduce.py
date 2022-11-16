import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from os import walk
import time
import os
from tqdm.auto import tqdm

# choosing the GPU number that is utilised
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def dataread(path):
    d = []
    f = []
    
    for (dir_path, dir_names, file_names) in walk(path):
        # gets rid of any pesky leftover .ipynb_checkpoints files
        if not dir_names == []:
            if not dir_names[0].startswith("."):
                f.extend(file_names)
                d.extend(dir_names)
    print(d)
    return d

name = ["HGG/","LGG/"]

for i in range(len(name)):
    path = "Brats_2018_data/Brats_2018_data/" + name[i]

    d = dataread(path)

    output_size = len(d)
    print("Reduce Data Size: " + path)
    
    data_Plot = nib.load(path + d[0] + "/" + d[0] + "_whimg_norm.nii.gz")
    print("Example current position [0] data type = ", data_Plot.get_fdata().dtype())
    print("Data type to convert to = ", data_Plot.get_fdata().astype(np.float32).dtype())
    print("Press Enter to continue")
    input(" ")

    for x in tqdm(range(output_size)):

        data_Plot = nib.load(path + d[x] + "/" + d[x] + "_whimg_norm.nii.gz")
        input_2 = data_Plot.get_fdata().astype(np.float32)
        data_out = nib.Nifti1Image(input_2, header=data_Plot.header, affine=data_Plot.affine)
        data_out.header.set_data_dtype(np.float32)                    
        nib.save(data_out, os.path.join(path + d[x] + "/" + d[x] +'_whimg_norm.nii.gz')) 
        