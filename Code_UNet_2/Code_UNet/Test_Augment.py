from Net_modules.Loading_data import Load_Dataset
import os
import Net_modules.Parameters_SEG as Param
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ideally run this with augment param set to false so that the original images are only transformed once but works either way.

KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=2):
    image = image.copy()

    keypoints = [
                (keypoints[1],keypoints[0]),
                (keypoints[3],keypoints[2]),
                (keypoints[5],keypoints[4]),
                (keypoints[7],keypoints[6])
            ]

    plt.figure(figsize=(8, 8))
    x_val = [x[0] for x in keypoints]
    y_val = [x[1] for x in keypoints]
    plt.scatter(x_val,y_val)
    plt.axis('off')
    plt.imshow(image)

Full_Path = os.getcwd() + "/" + "/Brats_2018/" #Param.Parameters.PRANO_Net["Train_paths"]["Data_path"]
folder = np.loadtxt(Full_Path + "/Training_dataset.csv", delimiter=",",dtype=str)
image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

# folder = read_csv_paths
dataset = Load_Dataset(Full_Path,image_folder_in,masks_folder_in)

x_in, y_in = dataset.__getitem__(533)

if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == True:    
    y_plt = y_in.squeeze()
#     print(np.shape(y_in))
    
#     y_in = y_in*0.5

    print("Truth before Augmentation")
    vis_keypoints(x_in, y_plt, KEYPOINT_COLOR)

    D1 = np.asarray([[y_plt[1],y_plt[3]],
                     [y_plt[0],y_plt[2]]]) 
    D2 = np.asarray([[y_plt[5],y_plt[7]],
                     [y_plt[4],y_plt[6]]]) 

    plt.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
    plt.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')
    plt.show()

    print("YIN", y_in)
    ### Augmentation
    x_out, y_out = dataset.augmentation(x_in, y_in)

    y_plt = y_out.squeeze()
    
    print("Output after Augmentation")
    vis_keypoints(x_out, y_plt, KEYPOINT_COLOR)

    D1 = np.asarray([[y_plt[1],y_plt[3]],
                     [y_plt[0],y_plt[2]]]) 
    D2 = np.asarray([[y_plt[5],y_plt[7]],
                     [y_plt[4],y_plt[6]]]) 

    plt.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
    plt.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')

    plt.show()

    # plot outputs
    
if Param.Parameters.PRANO_Net["Hyperparameters"]["Regress"] == False:
    
    print("Truth before Augmentation")
    plt.figure(figsize=(8, 8))
    plt.imshow(x_in)
    plt.imshow(y_in > 0, cmap='jet', alpha=0.5)
    plt.show()
    
    ### Augmentation
    x_out, y_out = dataset.augmentation(x_in, y_in)
    
    print("Output after Augmentation")
    plt.figure(figsize=(8, 8))
    plt.imshow(x_out)
    plt.imshow(y_out > 0, cmap='jet', alpha=0.5)
    plt.show()
    
    #### so the issue that i am having here is that the prano measurements are not shrunk where the images are. i thought that i had covered this by making the bilinear measurement analysis on the new images rather than the old ones but apparently im wrong . . .