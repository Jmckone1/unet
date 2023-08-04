import numpy as np
from skimage.measure import label, regionprops
import os
from tqdm.auto import tqdm
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

# read in the dataset
# adapt the p-RANO measurements to produce bounding boxes insteasd of this
class BBox():
        
    def get_data_list(Path, Folder):
        print(os.getcwd() + Path + Folder)
        output_list = []
        for (dir_path, dir_names, file_names) in os.walk(os.getcwd() + Path + Folder):
            if not file_names == []:
                for file in file_names:
                    if not file[0].startswith("."):
#                         print(file)
                        output_list.append(file)
                        counter = len(output_list)

        return output_list
    
    def calc_Bounding_Box(Path):
        path = ""
        folder = []
        
        mage_dir = BBox.get_data_list(Path, "labelsTr/")

        if not os.path.exists(os.getcwd() + Path + "BBoxlabelsTr/"):
            os.makedirs(os.getcwd() + New_path + "BBoxlabelsTr/")
            
        output_array = np.empty([len(mage_dir),4])

            
        for dir_name in tqdm(range(len(mage_dir))):

            mask = nib.load(os.getcwd() + New_path + "labelsTr/" + mage_dir[dir_name])
            numpy_mask = mask.get_fdata()
            
            lbl_0 = label(numpy_mask) 
            props = regionprops(lbl_0)
            area = 0
            biggest_box = [0,0,0,0]
            for prop in props:
#                 print(prop)
#                 print('Found bbox', prop.bbox)
#                 print(prop.area)
                if prop.area > area:
                    area = prop.area
                    biggest_box = prop.bbox
#             cv2.rectangle(img_1, (biggest_box[1], biggest_box[0]), (biggest_box[3], biggest_box[2]), (255, 0, 0), 2)
            
#             plt.imshow(numpy_mask)
#             plt.show()
#             plt.imshow(img_1)
#             plt.show()
            
#             print(biggest_box)
            value = mage_dir[dir_name].split('_')
            value_1 = value[-1].split('.')
            value_2 = value[0] + "_" + value[1] + "_" + value[2] + "_" + value[3] + "_" + value_1[0]

            np.savez(os.getcwd() + Path + "BBoxlabelsTr/" + value_2, BBox=biggest_box)
            output_array[dir_name] = biggest_box
        print(output_array)
if __name__ == "__main__":
    New_path = "/Datasets/Brats_2018_4/"
    BBox.calc_Bounding_Box(New_path)