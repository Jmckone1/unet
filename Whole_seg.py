path = "HGG"
index = 75
filetype = ["t1","flair","t1ce","t2","seg"]
img_data = np.empty((4,240,240,155))
img_labels = np.empty((240,240,155))
label_val = 0
save_lbl = True
file_output_name="whseg"
f = []
d = []
for (dir_path, dir_names, file_names) in walk(path):
    f.extend(file_names)
    d.extend(dir_names)
    for name in dir_names:
      print(os.path.join(path, name))
      file_label = name + '/' + name + r"_" + filetype[4] + '.nii.gz'
      l_full_path = os.path.join(path, file_label)
      l_img = nib.load(l_full_path)
      img_labels[:,:,:] = (l_img.get_fdata())[:,:,:]
      if label_val != 0:
        img_labels = (img_labels == label_val).astype(float)
      else:
        img_labels = (~(img_labels == label_val)).astype(float)

      if save_lbl == True:
        Label_img_save = nib.Nifti1Image(img_labels, np.eye(4))
        nib.save(Label_img_save, os.path.join(path, name + '/' + name + r"_" + file_output_name  + '.nii.gz'))  

      #y = img_labels[:,:,int(index - 155*np.floor(index/155))]
      #import matplotlib.pyplot as plt
      #plt.imshow(y)
      #plt.show()
      #label = img_labels[:,:,int(index - 155*np.floor(index/155))]
