from torch import nn
import torch.nn.functional as F
# import torch

# the inital smoothness value was 1, i have reduced this to 0.1 to check the effectiveness or difference in this case. from what i have looked at the  smoothness values stop division by 0 and to assist in avoiding overfitting, "the larger the smooth value the closer the overall value is to 1"

# for this first run i used the built in sigmoid here, will comment out for the next set as i think the sigmoid is already in place? this would make sense to me at least. will have to rerun when gpu space allows (likely tomorrow).

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice