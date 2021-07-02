import numpy as np
import torch

class Penalty():
    def __init__(self, w1, w2):
        #weighting values for orthogonality and area constraints
        
        self.orth_weight = w1
        self.area_weight = w2
        
    def MSELossorthog(self, output, target):

        # convert prediction and truth to np data
        output_val = output.data.cpu().numpy()

        #target_val = target.data.Tensor.cpu().numpy()
        target_val = target.cpu().data.numpy()

        batch_size = output_val.shape[0]

        loss = 0

        for i in range(batch_size):
            # calculate the lengths of each of the lines (prediction = l1, l2) (truth = l3,l4)
            l1 = np.sqrt(np.square(output_val[i][1]-output_val[i][3]) + np.square(output_val[i][0]-output_val[i][2]))
            l2 = np.sqrt(np.square(output_val[i][5]-output_val[i][7]) + np.square(output_val[i][4]-output_val[i][6]))
            l3 = np.sqrt(np.square(target_val[i][1]-target_val[i][3]) + np.square(target_val[i][0]-target_val[i][2]))
            l4 = np.sqrt(np.square(target_val[i][5]-target_val[i][7]) + np.square(target_val[i][4]-target_val[i][6]))

            # calculate the slope of each of the lines
            m1 = (abs(output_val[i][1]/l1-output_val[i][3]/l1))/(abs(output_val[i][0]/l1-output_val[i][2]/l1)+0.1)
            m2 = (abs(output_val[i][5]/l2-output_val[i][7]/l2))/(abs(output_val[i][4]/l2-output_val[i][6]/l2)+0.1)

            # calculate the orthogonality/ perpendicularity with 0 being perpendicular, 100 being parallel
            orthog = abs(np.dot(m1,m2))

            # calculate the area of the detected object to try and balance out the lengths of the lines (one of  which always appears to be significantly shorter than that of the other which is often more correct
            output_area = l1*l2 
            target_area = l3*l4

            output_area_norm = output_area / (output_area+target_area)
            target_area_norm = target_area / (output_area+target_area)

            area_penalty = abs(output_area_norm - target_area_norm) * 100

            loss = loss + torch.mean((output[i] - target[i])**2) + (orthog * self.orth_weight) + (area_penalty * self.area_weight)

        return loss / batch_size

    #https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
    #https://hewjunwei.wordpress.com/2013/01/26/obb-generation-via-principal-component-analysis/
    #https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask