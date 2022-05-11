import numpy as np
import torch

class Penalty():
    def __init__(self, w1, w2):
        #weighting values for orthogonality and area constraints
        
        self.orth_weight = w1
        self.area_weight = w2

    def unit_vector(self, vector):
            return vector / np.linalg.norm(vector)
        
    def orthogonality_penalty(self, data_in):

        x_maj = [data_in[1],data_in[3]]
        x_min = [data_in[0],data_in[2]]
        y_maj = [data_in[5],data_in[7]]
        y_min = [data_in[4],data_in[6]]

        major = [y_maj[1] - y_maj[0], x_maj[1] - x_maj[0]]
        minor = [y_min[1] - y_min[0], x_min[1] - x_min[0]]

        orthog = abs(np.dot(self.unit_vector(major), self.unit_vector(minor)))

        return orthog
                
    def MSELossorthog(self, output, target):

        # convert prediction and truth to np data
        output_val = output.data.cpu().numpy()
        target_val = target.cpu().data.numpy()
        batch_size = output_val.shape[0]

        loss = 0

        for i in range(batch_size):

            loss = loss + torch.mean((output[i] - target[i])**2) + (self.orthogonality_penalty(output_val[i]) * self.orth_weight)

        return loss / batch_size

    #https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
    #https://hewjunwei.wordpress.com/2013/01/26/obb-generation-via-principal-component-analysis/
    #https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask