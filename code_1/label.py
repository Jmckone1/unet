from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import time

path = "Brats_2018 data"
index = 75
filetype = ["t1","flair","t1ce","t2","seg"]
img_data = np.empty((4,240,240,155))
img_labels = np.empty((240,240,155))
label_val = 0
save_lbl = True
file_output_name="whseg"
f = []
d = []

path_ext = ["/HGG","/LGG"]
output_path = ["/HGG_2","/LGG_2"]

for input_ in range(len(path_ext)):
    counter = 0
    for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
        f.extend(file_names)
        d.extend(dir_names)
        counter = counter + 1
    print(counter)
    if input_ == 0:
      HGG_len = counter

for num, name in enumerate(d):

  if num < HGG_len-1:
      ext = path_ext[0]
      out = output_path[0]
  else:
      ext = path_ext[1]
      out = output_path[1]

  print(os.path.join(path + ext, name))
  print(num)
  file_label = name + '/' + name + r"_" + filetype[4] + '.nii.gz'
  l_full_path = os.path.join(path + ext, file_label)
  l_img = nib.load(l_full_path)
  img_labels[:,:,:] = (l_img.get_fdata())[:,:,:]
  if label_val != 0:
    img_labels = (img_labels == label_val).astype(float)
  else:
    img_labels = (~(img_labels == label_val)).astype(float)

  if save_lbl == True:
    Label_img_save = nib.Nifti1Image(img_labels, np.eye(4))
    nib.save(Label_img_save, os.path.join(path + out, name + '/' + name + r"_" + file_output_name  + '.nii.gz'))  
