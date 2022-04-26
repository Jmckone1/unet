from matplotlib.path import Path
from torch import nn
import numpy as np
import torch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class Dice_Evaluation():
    
    def __init__(self):
        return 0
    
    def dice_score(prediction, truth):
        # clip changes negative vals to 0 and those above 1 to 1
        pred_1 = np.clip(prediction, 0, 1.0)
        truth_1 = np.clip(truth, 0, 1.0)

        # binarize
        pred_1 = np.where(pred_1 > 0.5, 1, 0)
        truth_1 = np.where(truth_1 > 0.5, 1, 0)

        # Dice calculation
        product = np.dot(truth_1.flatten(), pred_1.flatten())
        dice_num = 2 * product + 1
        pred_sum = pred_1.sum()
        label_sum = truth_1.sum()
        dice_den = pred_sum + label_sum + 1
        score = dice_num / dice_den

        if pred_1.sum() == 0 and truth_1.sum() > 2:
            score = 0

        return score

class Jaccard_Evaluation():
    
    def __init__(self):
        return 0

    def Obb(input_array):

        input_array = input_array.detach().cpu().numpy()

        input_data = np.array([(input_array[1], input_array[0]),
                               (input_array[5], input_array[4]), 
                               (input_array[3], input_array[2]), 
                               (input_array[7], input_array[6])])

        input_covariance = np.cov(input_data,y = None, rowvar = 0,bias = 1)

        v, vect = np.linalg.eig(input_covariance)
        tvect = np.transpose(vect)
        # use the inverse of the eigenvectors as a rotation matrix and
        # rotate the points so they align with the x and y axes
        rotate = np.dot(input_data,vect)

        # get the minimum and maximum x and y 
        mina = np.min(rotate,axis=0)
        maxa = np.max(rotate,axis=0)
        diff = (maxa - mina)*0.5

        # the center is just half way between the min and max xy
        center = mina + diff

        # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
        corners = np.array([center+[-diff[0],-diff[1]],
                            center+[ diff[0],-diff[1]],
                            center+[ diff[0], diff[1]],
                            center+[-diff[0], diff[1]],
                            center+[-diff[0],-diff[1]]])

        # use the the eigenvectors as a rotation matrix and
        # rotate the corners and the centerback
        corners = np.dot(corners,tvect)
        center = np.dot(center,tvect)

        return corners, center

    def mask(shape,corners):
        nx, ny = shape

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x,y)).T

        path = Path(corners)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))

        return grid