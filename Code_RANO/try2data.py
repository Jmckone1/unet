import numpy as np

import matplotlib.pyplot as plt

import nibabel as nib
from os import walk
import time

from skimage.morphology import label
#from skimage.measure import regionprops, mesh_surface_area, marching_cubes
#from skimage.morphology import label, binary_erosion, disk
#from skimage.measure import regionprops, mesh_surface_area, marching_cubes, find_contours
#from scipy import signal
#from scipy.spatial.distance import cdist
from collections import namedtuple

from deepneuro.utilities.conversion import read_image_files
#from Code_RANO.calc_RANO_2D import dataread
from Code_RANO.calc_RANO_2D import *

path = "Brats_2018_data_split/Training/HGG/"



for x in range(100):
    print(x)
    dataPLot = nib.load("Brats_2018_data_split/Training/HGG/Brats18_CBICA_ATF_1/Brats18_CBICA_ATF_1_whseg.nii.gz")
    input_2a = dataPLot.get_fdata()
    plt.imshow(input_2a[:,:,75])
    plt.show()

    #calc_2D_RANO_measure(input_2, pixdim=(240,240), affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath="output_rano", verbose=False)
