from skimage.morphology import label, binary_erosion, disk, binary_dilation
from deepneuro.utilities.conversion import read_image_files
from skimage.measure import regionprops, find_contours
from scipy.spatial.distance import cdist
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
import pandas as pd 
from os import walk
import numpy as np
import shutil
import time
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
class DeepNeuro():
    def _get_pixdim(pixdim, affine, input_affine, verbose=True):

        """ Currently only functional for 3D images.
        """

        if pixdim is None:
            if affine is not None:
                pixdim = np.abs(affine.diagonal()[0:-1])
            elif input_affine is not None:
                pixdim = np.abs(input_affine.diagonal()[0:-1])
            else:
                if verbose:
                    print('Warning -- no resolution provided. Assuming isotropic.')
                return [1, 1, 1]

        return pixdim

    class Point(namedtuple('Point', 'x y')):

        __slots__ = ()
        @property
        def length(self):
            return (self.x ** 2 + self.y ** 2) ** 0.5

        def __sub__(self, p):
            return Point(self.x - p.x, self.y - p.y)

        def __str__(self):
            return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)

    def plot_contours(contours, lw=4, alpha=0.5):
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha)

    def vector_norm(p):
        length = p.length
        return DeepNeuro.Point(p.x / length, p.y / length)

    def compute_pairwise_distances(P1, P2, min_length=10):
        euc_dist_matrix = cdist(P1, P2, metric='euclidean')
        indices = []
        for x in range(euc_dist_matrix.shape[0]):
            for y in range(euc_dist_matrix.shape[1]):

                p1 = DeepNeuro.Point(*P1[x])
                p2 = DeepNeuro.Point(*P1[y])
                d = euc_dist_matrix[x, y]

                if p1 == p2:
                    continue
                elif d < min_length:
                    continue
                else:
                    indices.append([p1, p2, d])

        return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)

    def interpolate(p1, p2, d):

        X = np.linspace(p1.x, p2.x, round(d)).astype(int)
        Y = np.linspace(p1.y, p2.y, round(d)).astype(int)
        XY = np.asarray(list(set(zip(X, Y))))
        return XY

    def find_largest_orthogonal_cross_section(pairwise_distances, img, tolerance=0.01):
        for i, (p1, p2, d1) in enumerate(pairwise_distances):

            # Compute intersections with background pixels
            XY = DeepNeuro.interpolate(p1, p2, d1)
            intersections = sum(img[x, y] == 0 for x, y in XY)
            if intersections == 0:

                V = DeepNeuro.vector_norm(DeepNeuro.Point(p2.x - p1.x, p2.y - p1.y))

                # Iterate over remaining line segments
                for j, (q1, q2, d2) in enumerate(pairwise_distances[i:]):

                    W = DeepNeuro.vector_norm(DeepNeuro.Point(q2.x - q1.x, q2.y - q1.y))
                    if abs(np.dot(V, W)) < tolerance:

                        XY = DeepNeuro.interpolate(q1, q2, d2)
                        intersections = sum(img[x, y] == 0 for x, y in XY)
                        if intersections == 0:
                            return [p1, p2, q1, q2],[d1,d2]

    def calc_2D_RANO_measure(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath=None, verbose=True):

        input_data, input_affine = read_image_files(input_data, return_affine=True)
        pixdim = DeepNeuro._get_pixdim(pixdim, affine, input_affine)
        Bidimensional_output=np.zeros([input_data.shape[2],8])

        max_2ds = []
        major_diameter = None

        for z_slice in range(input_data.shape[2]): # here it wants to be not all the objects but the largest of the objects
            connected_components= label(input_data[:,:,z_slice], connectivity=1)

            (unique, counts) = np.unique(connected_components, return_counts=True)
            counts[0] = 0
            largest_component = (connected_components == np.argmax(counts)).astype(int)
            if sum(sum(largest_component)) > 10:
                label_slice = binary_dilation(largest_component).astype(largest_component.dtype)
                label_properties = regionprops(label_slice)
                current_major = label_properties[0].major_axis_length
                current_orientation = label_properties[0].orientation

                if np.sum(label_slice) == 0:
                    continue

                p,d = DeepNeuro.calc_rano_points(label_slice)

                if p != None:
                    x_dim = abs(np.cos(current_orientation) * current_major)
                    y_dim = abs(np.sin(current_orientation) * current_major)
                    current_major = ((x_dim * pixdim[0])**2 + (y_dim * pixdim[1])**2)**.5

                    if major_diameter is None:
                        major_diameter = current_major
                    elif current_major > major_diameter:
                        major_diameter = current_major

                    if p[0] != 0:
                        data_out = []
                        for i in range(4):
                            data_out = np.append(data_out, p[i].x)
                            data_out = np.append(data_out, p[i].y)
                    else:
                        data_out = [0,0,0,0,0,0,0,0]

                    Bidimensional_output[z_slice,:] = data_out

                    if major_diameter is not None:
                        max_2ds += [major_diameter]

        return Bidimensional_output

    def calc_rano_points(binary_image, tol=0.01, output_file=None, background_image=None, verbose=False):

        """ Code developed by James Brown, postdoctoral fellow at the QTIM lab.
        """

        # Estimate lesion height and width
        # Dilate slightly to prevent self-intersections, and compute contours
        dilated = binary_erosion(binary_image, disk(radius=1)).astype('uint8') * 255
        contours = find_contours(dilated, level=1)

        if len(contours) == 0:
            if verbose:
                print("No lesion contours > 1 pixel detected.")
            return [0,0,0,0],[0,0]

        # Calculate pairwise distances over boundary
        outer_contour = np.round(contours[0]).astype(int)  # this assumption should always hold...

        #euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=width)
        euc_dist_matrix, ordered_diameters = DeepNeuro.compute_pairwise_distances(outer_contour, outer_contour, min_length=10)

        # Exhaustive search for longest valid line segment and its orthogonal counterpart
        try:
            q = [0,0,0,0]
            d = [0,0]
            q,d = DeepNeuro.find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
        except TypeError:
            if verbose:
                print("Error: unable to compute RANO measurement")
            return [0,0,0,0],[0,0]
        return q,d

    def dataread(path):
        d = []

        for (dir_path, dir_names, file_names) in walk(path):
            # gets rid of any pesky leftover .ipynb_checkpoints files
            if not dir_names == []:
                for name in range(len(dir_names)):
                    if not dir_names[name].startswith("."):
                        d.append(dir_names[name])
        return d

class Pre_process():
    def __init__(self):
        print("Init")
        
    def calc_Bi_Linear(New_path):
        print("Calcualting Bi-linear measurements of the input slices")
        
        image_directories = Pre_process.get_data_list(New_path, "labelsTr/")

        if not os.path.exists(os.getcwd() + New_path + "BiLabelsTr/"):
            os.makedirs(os.getcwd() + New_path + "BiLabelsTr/")

        for dir_name in tqdm(image_directories):

            mask = nib.load(os.getcwd() + New_path + "labelsTr/" + dir_name)
            numpy_mask = mask.get_fdata()

            size = numpy_mask.ndim
            
            if size == 2:
                numpy_mask = numpy_mask[:,:,np.newaxis]
                size = numpy_mask.ndim

            Bi_output = DeepNeuro.calc_2D_RANO_measure(numpy_mask, pixdim=(numpy_mask.shape[0],numpy_mask.shape[1]), affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath="output_rano", verbose=False)
            
            for image_slice in range(np.shape(numpy_mask)[size-1]):
                np.savez(os.getcwd() + New_path + "BiLabelsTr/" + dir_name[:-7], RANO=Bi_output[image_slice,:])
                
    def get_data_list(Path, Folder):
        
        output_list = []
        for (dir_path, dir_names, file_names) in os.walk(os.getcwd() + Path + Folder):
            if not file_names == []:
                for file in file_names:
                    if not file[0].startswith("."):
                        output_list.append(file)
                        counter = len(output_list)

        return output_list
        
    def reformat_dataset(Old_path, New_path, folds_value = 10, saveFile = False, saveCSV = False, saveBilinear = False, resize_axis = 1, debug = False):
        print("Reformatting dataset")
        
        # we need to split the data into the following format: 
            # [1,512,512,250] -> [1,512,512]
        # each one needs to be saved into the same foramt, we shall stick to nii.gz for 
        # the time being though in many cases the dataset is saved into .png or .jpg 
        # instead so i will need to test if this makes it faster, smaller sizes to store 
        # or whether it is just personal preference. My first assumption would be that 
        # it is quicker.
        
        image_directories = Pre_process.get_data_list(Old_path, "imagesTr/")
        num_of_slices = 0
        image_csv_append = []
        masks_csv_append = []
        
        if not os.path.exists(os.getcwd() + New_path):
            print(os.getcwd() + New_path, "Save directory does not yet exist")
            os.makedirs(os.getcwd() + New_path)
            os.makedirs(os.getcwd() + New_path + "imagesTr/")
            os.makedirs(os.getcwd() + New_path + "labelsTr/")
            print(os.getcwd() + New_path, "Save directory created")
        else:
            print("Save directory already exists. Overwriting . . . ")
        
        for dir_name in tqdm(range(len(image_directories))):
            
            # load the training label and iamge from original path file structure
            image = nib.load(os.getcwd() + Old_path + "imagesTr/" + image_directories[dir_name])
            label = nib.load(os.getcwd() + Old_path + "labelsTr/" + image_directories[dir_name])
            
            # extract the nifty data
            numpy_image = image.get_fdata()
            numpy_label = label.get_fdata()
            
            # get the number of channels that the data has
            size = numpy_image.ndim
           
            for image_slice in tqdm(range(np.shape(numpy_image)[size-1])):
                num_of_slices+=1
                if saveFile == True:
                    # convert to numpy array 
                    image_to_save = numpy_image[:,:,image_slice]
                    label_to_save = numpy_label[:,:,image_slice]

                    image_to_save = Pre_process.resize(image_to_save,resize_axis)
                    if debug == True: print("Image shape: ", np.shape(image_to_save))
                    label_to_save = Pre_process.resize(label_to_save,resize_axis)
                    if debug == True: print("Label shape: ", np.shape(label_to_save))

                    # give each newly constructed nifty file a header that will adapt to the data
                    empty_header = nib.Nifti1Header()
                    new_image = nib.Nifti1Image(image_to_save, affine=image.affine, header=empty_header)
                    new_label = nib.Nifti1Image(label_to_save, affine=label.affine, header=empty_header)

                    # save each slice as an image to the new folder that is created here.
                    nib.save(new_image, os.getcwd() + New_path + "imagesTr/" + image_directories[dir_name][:-7] + "_" + str(image_slice))
                    nib.save(new_label, os.getcwd() + New_path + "labelsTr/" + image_directories[dir_name][:-7] + "_" + str(image_slice))

                image_csv_append.append(image_directories[dir_name][:-7] + "_" + str(image_slice))
                masks_csv_append.append(image_directories[dir_name][:-7] + "_" + str(image_slice))
        # finally follow up by creating the csv with the correct fold allocation and 
        # index value for each example. this will likely require using the original
        # folder as a reference for the files to makes sure each specific volume is not]
        # split up amongst the validation splits.
        
        fold = np.zeros(num_of_slices).astype(int)
        for loop in range(folds_value):
            fold[int(num_of_slices/folds_value)*loop:int(num_of_slices/folds_value)*(loop+1)] = loop
        if int(np.ceil((num_of_slices / folds_value)%int(num_of_slices / folds_value)*(folds_value+1))) != 0:
            fold[-int(np.ceil((num_of_slices / folds_value)%int(num_of_slices / folds_value)*(folds_value+1))):] = folds_value -1

        
        if saveCSV == True:
            Pre_process.create_csv(image_csv_append, masks_csv_append, fold, os.getcwd() + New_path + "Training_dataset.csv")
        
        if saveBilinear == True:
            Pre_process.calc_Bi_Linear(New_path)
            
    def reformat_dataset_brats(Old_path, New_path, folds_value = 10, saveFile = False, saveCSV = False, saveBilinear = False, resize_axis = 1, debug = True, dir_search = False):
        print("Reformatting dataset")
        # i want to track the HGG or LGG allocation but cant wrap my head around it 
        # at the moment so i will just leave this here as a note.
        
        # we need to split the data into the following format: 
            # [1,512,512,250] -> [1,512,512]
        # each one needs to be saved into the same foramt, we shall stick to nii.gz for 
        # the time being though in many cases the dataset is saved into .png or .jpg 
        # instead so i will need to test if this makes it faster, smaller sizes to store 
        # or whether it is just personal preference. My first assumption would be that 
        # it is quicker.
        
        image_directories = Pre_process.get_data_list(Old_path, "")
        if image_directories == []:
            print("Image directory is empty, please check path.")
            return
        
        num_of_slices = 0
        image_csv_append = []
        masks_csv_append = []
        old_path_list = []
        
#         image_search_string = "_flair_norm.nii.gz"
        image_search_string = "_whimg_norm.nii.gz"
        label_search_string = "_whseg_norm.nii.gz"
        
        image_directory = []
        label_directory = []
        
        for i in range(len(image_directories)):
            if image_directories[i].endswith(image_search_string):
                image_directory.append(image_directories[i])
            if image_directories[i].endswith(label_search_string):
                label_directory.append(image_directories[i])

        if not os.path.exists(os.getcwd() + New_path):
            print(os.getcwd() + New_path, "Save directory does not yet exist")
            os.makedirs(os.getcwd() + New_path)
            os.makedirs(os.getcwd() + New_path + "imagesTr/")
            os.makedirs(os.getcwd() + New_path + "labelsTr/")
            print(os.getcwd() + New_path, "Save directory created")
        else:
            print("Save directory already exists. Overwriting . . . ")
        
        for dir_name in tqdm(range(len(image_directory))):
            # load the training label and iamge from original path file structure
            image = nib.load(os.getcwd() + Old_path + image_directory[dir_name][:-len(image_search_string)] + "/" + image_directory[dir_name])
            
            label = nib.load(os.getcwd() + Old_path + label_directory[dir_name][:-len(label_search_string)] + "/" + label_directory[dir_name])
            
            
            # extract the nifty data
            numpy_image = image.get_fdata()
            numpy_label = label.get_fdata()
#             print("Input image shape 1", np.shape(numpy_image))
#             print("Input label shape 1", np.shape(numpy_label))
            # get the number of channels that the data has
            size = numpy_image.ndim
            
            for image_slice in tqdm(range(np.shape(numpy_image)[size-1])):
                num_of_slices+=1
                if saveFile == True:
                    # convert to numpy array
                    if numpy_image.ndim == 3:
                        image_to_save = numpy_image[:,:,image_slice]
                    if numpy_image.ndim == 4:
                        image_to_save = numpy_image[:,:,:,image_slice]
                    label_to_save = numpy_label[:,:,image_slice]
                    
                    image_to_save = Pre_process.resize(image_to_save,resize_axis)
                    label_to_save = Pre_process.resize(label_to_save,resize_axis)

                    # give each newly constructed nifty file a header that will adapt to the data
                    empty_header = nib.Nifti1Header()
                    new_image = nib.Nifti1Image(image_to_save, affine=image.affine, header=empty_header)
                    new_label = nib.Nifti1Image(label_to_save, affine=label.affine, header=empty_header)
               
                    # param doesnt exist otherwise this would be a good idea . . .
#                     if Param.Parameters.PRANO_Net["Global"]["Debug"]: 
#                         print(os.getcwd() + New_path + "imagesTr/" + image_directory[dir_name][:-7] + "_" + str(image_slice) + ".nii.gz")
                    # save each slice as an image to the new folder that is created here.
                    nib.save(new_image, os.getcwd() + New_path + "imagesTr/" + image_directory[dir_name][:-7] + "_" + str(image_slice) + ".nii.gz")
                    nib.save(new_label, os.getcwd() + New_path + "labelsTr/" + label_directory[dir_name][:-7] + "_" + str(image_slice) + ".nii.gz")

                image_csv_append.append(image_directory[dir_name][:-7] + "_" + str(image_slice))
                masks_csv_append.append(label_directory[dir_name][:-7] + "_" + str(image_slice))
                old_path_list.append(Old_path)
                
        # finally follow up by creating the csv with the correct fold allocation and 
        # index value for each example. this will likely require using the original
        # folder as a reference for the files to makes sure each specific volume is not]
        # split up amongst the validation splits.
#         print("Num of slice", num_of_slices)
        fold = np.zeros(num_of_slices).astype(int)
#         print(np.shape(fold))
        for loop in range(folds_value):
#             print("loopnumber", loop)
#             print(int(num_of_slices/folds_value)*loop,
#                   "-",
#                   int(num_of_slices/folds_value)*(loop+1),
#                   "=",
#                   loop)
            fold[int(num_of_slices/folds_value)*loop:int(num_of_slices/folds_value)*(loop+1)] = loop
#             print("NUM set to 9: -", int(np.ceil((num_of_slices / folds_value)%int(num_of_slices / folds_value)*(folds_value+1))))
#         print("fold before", np.unique(fold))
        if int(np.ceil((num_of_slices / folds_value)%int(num_of_slices / folds_value)*(folds_value+1))) != 0:
            fold[-int(np.ceil((num_of_slices / folds_value)%int(num_of_slices / folds_value)*(folds_value+1))):] = folds_value -1
#         print("fold after", np.unique(fold))
        
        if saveCSV == True:
            Pre_process.create_csv(image_csv_append, masks_csv_append,  fold, os.getcwd() + New_path + "Training_dataset.csv")
        
        if saveBilinear == True:
            Pre_process.calc_Bi_Linear(New_path)
        return [image_csv_append, masks_csv_append, fold, os.getcwd() + New_path + "Training_dataset.csv", old_path_list]

    def create_csv(image_directory_names, mask_directory_names, fold, save_directory, old_path_list = []):
        print("Creating csv")
        # we need to allocate the index for the images here and then also apply an
        # additional index for the 5 fold cross validation that we can perform. 
        # This would of course be less customisable in the moment but would instead maybe
        # require the values to be indicated at this point for the amount of folds and how
        # large in quantity they are based on the dataset size. each line for the csv 
        # would require the following format:
            # [image_path, index, fold]
        # we could store this in a dictionary or a csv format (csv may be better here 
        # since there is a large amount of repetition.
        
        indexes = range(len(image_directory_names))
        
        if old_path_list == []:
            old_path_list = np.zeros(np.size(indexes))
        
        stored_values = np.stack((image_directory_names, mask_directory_names, indexes, fold, old_path_list), -1)
        
        pd.DataFrame(stored_values).to_csv(save_directory, header=None, index=None)
        
    def normalisation():
        print("Normalising dataset")
        # Normalisation of CT data is different to that of an MRI dataset. 
        # So might as well put it in here.
        # I could also (AT A LATER DATE) put the standardisation functions here for 
        # the MRI data. . .
        
    def resize(image,size_multiplier):
        
        sizes = np.shape(image)
#         print("Mult", size_multiplier)
        if size_multiplier == 1:
            resized_image = image
        else:
            resized_image = cv2.resize(image, dsize=(int(sizes[0]*size_multiplier), 
                                       int(sizes[1]*size_multiplier)), interpolation=cv2.INTER_AREA)

        return resized_image
        
if __name__ == "__main__":

    
    preproc_CT = False
    preproc_Brats = False
    preproc_small = True
    
    if preproc_CT == True:
        Old_path = "/2023_Lung_CT_code/Task06_Lung/"
        New_path = "/CT_Dataset/Task06_Lung/"
        Pre_process.reformat_dataset(Old_path,
                                     New_path, 
                                     folds_value = 10, 
                                     saveFile = False, 
                                     saveCSV = False, 
                                     saveBilinear = True,
                                     resize_axis = 0.5)

#     Pre_process.calc_Bi_Linear(New_path)

    if preproc_Brats == True:
        Old_paths = ["/Brats_2018_data/Brats_2018_data/HGG/", "/Brats_2018_data/Brats_2018_data/LGG/"]
        New_path = "/Brats_2018/"
    #     Old_paths = ["/Data_1/HGG/", "/Data_1/LGG/"]
    #     New_path = "/brats_Dataset/"

        data_out = [[],[],[],[],[]]
        image_csv_data = []
        masks_csv_data = []
        folds = []
        new_dir = []
        old_dir = []

        for input_path in Old_paths:
            data_out = Pre_process.reformat_dataset_brats(input_path,
                                                                  New_path, 
                                                                  folds_value = 10, 
                                                                  saveFile = False, 
                                                                  saveCSV = False, 
                                                                  saveBilinear = True,
                                                                  resize_axis = 1, 
                                                                  debug = False)
            image_csv_data.extend(data_out[0])
            masks_csv_data.extend(data_out[1])
            folds.extend(data_out[2])
            new_dir.extend(data_out[3])
            old_dir.extend(data_out[4])
        Pre_process.create_csv(image_csv_data, masks_csv_data, folds, os.getcwd() + New_path + "Training_dataset.csv" , old_dir)
    
    if preproc_small == True:
        Old_paths = ["/Data_1/HGG/", "/Data_1/LGG/"]
        New_path = "/Brats_2018_small_4/"
    #     Old_paths = ["/Data_1/HGG/", "/Data_1/LGG/"]
    #     New_path = "/brats_Dataset/"

        data_out = [[],[],[],[],[]]
        image_csv_data = []
        masks_csv_data = []
        folds = []
        new_dir = []
        old_dir = []

        for input_path in Old_paths:
            data_out = Pre_process.reformat_dataset_brats(input_path,
                                                                  New_path, 
                                                                  folds_value = 10, 
                                                                  saveFile = True, 
                                                                  saveCSV = False, 
                                                                  saveBilinear = True,
                                                                  resize_axis = 1, 
                                                                  debug = False)
            image_csv_data.extend(data_out[0])
            masks_csv_data.extend(data_out[1])
            folds.extend(data_out[2])
            new_dir.extend(data_out[3])
            old_dir.extend(data_out[4])
        Pre_process.create_csv(image_csv_data, masks_csv_data, folds, os.getcwd() + New_path + "Training_dataset.csv" , old_dir)
#     Pre_process.calc_Bi_Linear(New_path)

#     image_directories = Pre_process.get_data_list(New_path, "imagesTr/")
#     num_of_slices = 0
#     csv_append = []

#     for dir_name in tqdm(image_directories):
#         # print(os.getcwd() + New_path + "imagesTr/" + dir_name)

#         # load the training label and iamge from original path file structure
#         image = nib.load(os.getcwd() + New_path + "imagesTr/" + dir_name)
#         label = nib.load(os.getcwd() + New_path + "labelsTr/" + dir_name)

#         # extract the nifty data
#         numpy_image = image.get_fdata()
#         numpy_label = label.get_fdata()
        
#         print(np.shape(numpy_image))
        
#         plt.imshow(numpy_image)
#         plt.show()
#         plt.imshow(numpy_label)
#         plt.show()
        
#         input("")
