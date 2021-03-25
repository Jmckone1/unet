import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import label
from skimage.measure import regionprops
from skimage.measure import regionprops_table

def calc_RANO(input_label, affine=None, resolution_x=1, resolution_y=1, resolution_z=1, output_csv=None, image_data=None, output_image_filepath=None, display_image=False):

    """ Calculate RANO criteria. Assumes data is oriented [x, y, z]. TODO: Make dimension agnostic.
        Code modified from original written by Ken Chang.
        My editing of the code is not great; TODO: Refactor.
    """

    rano_measures, rano_slices, rano_props = [], [], []

    connected_components = label(input_label, connectivity=2)
    component_labels = np.unique(connected_components)
    
    print("labels",component_labels)
    
    # each lesion object detected throughout all slices i think
    for lesion_idx in component_labels:

        lesion = connected_components.astype(int)
        lesion[connected_components != lesion_idx] = 0
        
        major_diameter, minor_diameter, rano_slice, region_props = [None] * 4

        volume_threshold = 2 * resolution_z
        if volume_threshold < 10:
            volume_threshold = 10
        
        # for each slice
        for z_slice in range(lesion.shape[2]):

            lesion_slice = lesion[..., z_slice]

            if np.sum(lesion_slice) == 0:
                continue

            lesion_properties = regionprops(lesion_slice)
            
            lesion_print = regionprops_table(lesion_slice)
            print(lesion_print["bbox-1"])
            input(" ")
            
            current_major = lesion_properties[0].major_axis_length * resolution_x
            current_minor = lesion_properties[0].minor_axis_length * resolution_y

            if current_major < volume_threshold:
                continue
            if major_diameter is None:
                major_diameter, minor_diameter, rano_slice, region_props = current_major, current_minor, z_slice, lesion_properties
            elif current_major > major_diameter:
                major_diameter, minor_diameter, rano_slice, region_props = current_major, current_minor, z_slice, lesion_properties

        if major_diameter is not None:
            rano_measures += [major_diameter * minor_diameter]
            rano_slices += [rano_slice]
            rano_props += [region_props]

    if len(rano_measures) < 5:
        sum_rano = np.sum(rano_measures)
    else:
        sum_rano = np.sum(rano_measures.sort()[-5:])

    if output_csv is not None:
        if not os.path.exists(output_csv):
            pass

    if output_image_filepath is not None or display_image:

        for idx, z_slice in enumerate(rano_slices):
            print("index",idx)
            lesion_props = rano_props[idx][0]

            if image_data is None:
                display_data = input_label[..., z_slice]
            else:
                display_data = image_data[..., z_slice]

            center_y, center_x = lesion_props.centroid
            major_angle = lesion_props.orientation

            minor_angle = major_angle + np.pi / 2

            half_major, half_minor = lesion_props.major_axis_length / 2, lesion_props.minor_axis_length / 2

            major_x_1 = center_x + np.cos(major_angle) * half_minor
            major_y_1 = center_y - np.sin(major_angle) * half_minor
            major_x_2 = center_x - np.cos(major_angle) * half_minor
            major_y_2 = center_y + np.sin(major_angle) * half_minor

            minor_x_1 = center_x + np.cos(minor_angle) * half_major
            minor_y_1 = center_y - np.sin(minor_angle) * half_major
            minor_x_2 = center_x - np.cos(minor_angle) * half_major
            minor_y_2 = center_y + np.sin(minor_angle) * half_major

            plt.imshow(display_data, interpolation='none', origin='lower', cmap='gray')
            plt.plot(center_x, center_y, 'ro') 
            plt.plot(major_x_1, major_y_1, 'go') 
            plt.plot(major_x_2, major_y_2, 'bo')
            plt.plot(minor_x_1, minor_y_1, 'yo') 
            plt.plot(minor_x_2, minor_y_2, 'co') 
            plt.show()
            
            #plt.plot(center_x, center_y, 'ro') 
            #plt.plot(major_x_1, major_y_1, 'go') 
            #plt.plot(major_x_2, major_y_2, 'bo')
            #plt.plot(minor_x_1, minor_y_1, 'yo') 
            #plt.plot(minor_x_2, minor_y_2, 'co') 
            #plt.show()
            
            # the data that i want to parameterise is the following:
            # x, y, len_x, len_y, orientation and crossover
            

    return sum_rano

import nibabel as nib
from os import walk
d = []
f = []
for (dir_path, dir_names, file_names) in walk("Brats_2018_data_split/Validation/HGG/"):
                # gets rid of any pesky leftover .ipynb_checkpoints files
                if not dir_names == []:
                    if not dir_names[0].startswith("."):
                        
                        f.extend(file_names)
                        d.extend(dir_names)
                        
print(d)

for i in range(len(d)):                        
    input_1 = l_img = nib.load("Brats_2018_data_split/Validation/HGG/" + d[i] + "/" + d[i] + "_whseg.nii.gz")
    input_1 = (input_1.get_fdata())[:,:,:]

    j = calc_RANO(input_1, display_image=True)

    print(j) # prints the sum_RANO
    
#"Brats_2018_data_split/Validation/HGG/Brats18_2013_21_1/Brats18_2013_21_1_whseg.nii.gz"