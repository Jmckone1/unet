from torch.utils.data.dataset import Dataset
from os import walk
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import time
import random
import torchvision.transforms.functional as TF
import torchvision
random.seed(0)
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class Jaccard_Index():
    def __init__(self):
        return 0
    
    def input_data():
        path_ext = ["/HGG","/LGG"]
        path ="Brats_2018_data_split/Validation"
        d = []
        index_max = []
        index_max.extend([0])
        f = []

        path_ext = path_ext

        c_s = 0

        # each extension - HGG or LGG
        for input_ in range(len(path_ext)):
            counter = 0
            # each folder in extension
            for (dir_path, dir_names, file_names) in walk(path + path_ext[input_]):
                # gets rid of any pesky leftover .ipynb_checkpoints files
                if not dir_names == []:
                    if not dir_names[0].startswith("."):

                        d.extend(dir_names)

                        counter = len(d)
                        #print("D:",len(d))

            for directory in range(counter-c_s):
                if directory == 0:
                    if input_ == 0:
                        c_s = counter
                if input_ == 1:
                    directory = directory + c_s

                file = d[directory] + '/' + d[directory] + "r_" + "whseg" + '.nii.gz'
                full_path = os.path.join(path + path_ext[input_], file)
                img_a = nib.load(full_path)
                img_data = img_a.get_fdata()

                #print(img_data.shape)
                index_max.extend([img_data.shape[2] + index_max[-1]])
                for xc in range(img_data.shape[2]):
                    f.extend([np.sum(np.sum(img_data[:,:,xc]))])

                #print(index_max)
                # value for extension swapping
                if input_ == 0:
                    HGG_len = index_max[-1]
            #print(HGG_len)
            #print(index_max)


        # inputs to global
        count = index_max[-1] # anything with 155 in it needs to be redone to not rely on the hard coded value

        return f

    def MSELossorthog(output, target):

        output_val = output.data.cpu().numpy()

        l1 = np.sqrt(np.square(output_val[1]-output_val[3]) + np.square(output_val[0]-output_val[2]))
        l2 = np.sqrt(np.square(output_val[5]-output_val[7]) + np.square(output_val[4]-output_val[6]))

        m1 = (abs(output_val[1]/l1-output_val[3]/l1))/(abs(output_val[0]/l1-output_val[2]/l1)+0.1)
        m2 = (abs(output_val[5]/l2-output_val[7]/l2))/(abs(output_val[4]/l2-output_val[6]/l2)+0.1)

        orthog = abs(np.dot(m1,m2))

        weight = 2

        loss = torch.mean((output - target)**2) + (orthog * weight)
        return loss

    #https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
    #https://hewjunwei.wordpress.com/2013/01/26/obb-generation-via-principal-component-analysis/
    #https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

    # image interpolation multiplier
    size = 1

    # BCE with Logits loss, may change to soft dice
    criterion = MSELossorthog

    n_epochs = 6
    input_dim = 4
    label_dim = 8
    hidden_dim = 32

    display_step = False
    batch_size = 16
    lr = 0.0002
    initial_shape = int(240 * size)
    target_shape = int(8)
    device = 'cuda'

    def Obb(input_array):

        input_array = input_array.detach().cpu().numpy()

        input_data = np.array([(input_array[1], input_array[0]),
                               (input_array[5], input_array[4]), 
                               (input_array[3], input_array[2]), 
                               (input_array[7], input_array[6])])

        input_covariance = np.cov(input_data,y = None, rowvar = 0,bias = 1)

        v, vect = np.linalg.eig(input_covariance)
        tvect = np.transpose(vect)
        #use the inverse of the eigenvectors as a rotation matrix and
        #rotate the points so they align with the x and y axes
        rotate = np.dot(input_data,vect)

        # get the minimum and maximum x and y 
        mina = np.min(rotate,axis=0)
        maxa = np.max(rotate,axis=0)
        diff = (maxa - mina)*0.5

        # the center is just half way between the min and max xy
        center = mina + diff

        #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
        corners = np.array([center+[-diff[0],-diff[1]],
                            center+[ diff[0],-diff[1]],
                            center+[ diff[0], diff[1]],
                            center+[-diff[0], diff[1]],
                            center+[-diff[0],-diff[1]]])

        #use the the eigenvectors as a rotation matrix and
        #rotate the corners and the centerback
        corners = np.dot(corners,tvect)
        center = np.dot(center,tvect)

        return corners, center

    def mask(shape,corners):
        nx, ny = shape

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x,y)).T

        path = Path(corners)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))

        return grid