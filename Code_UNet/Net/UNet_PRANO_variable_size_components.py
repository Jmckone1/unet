import torch
from torch import nn
import Net_modules.Parameters_PRANO as Param
import numpy as np 

class Contract(nn.Module):
    def __init__(self, input_channels):
        super(Contract, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        
        x = self.maxpool(x)
        
        return x
    
class FeatureMap(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMap, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class UNet(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=32, depth=5):
        super(UNet, self).__init__()
        self.depth = depth
        
        self.model = []
        self.model.append(FeatureMap(input_channels, hidden_channels))
        width = 1
        for d in range(self.depth):
            self.model.append(Contract(hidden_channels * width))
            width = width + width
        self.model.append(nn.Linear(((hidden_channels*width)*
                                  int(Param.Parameters.PRANO_Net
                                      ["Hyperparameters"]["Image_size"][0]/
                                      width)*
                                  int(Param.Parameters.PRANO_Net
                                      ["Hyperparameters"]["Image_size"][1]/
                                      width)), output_channels) )
        print(self.model)
    def forward(self, data_in):
        
        if(data_in.ndim == 3):
            data_in = data_in[:,np.newaxis,:,:]
            
        contract = self.model[0](data_in)
        for d in range(self.depth):
            contract = self.model[d](contract)

        data_flat = torch.flatten(contract, start_dim=1)
        data_out = self.Linear1(data_flat)

        return data_out

# =============================================================================
# # output testing for the model and saving of the model diagram usign graphiz
# # 4 input MRI sequences, 1 segmentation output
# # 16 hidden layers for this model
# input_dim = 4
# label_dim = 8
# hidden_dim = 16
# 
# device = 'cuda'
#                       
# # to use the model with weights, frozen or unfrozen utilise UNet.load_weights
# # to use the model without having any weights use UNet without allow_update and path
# model = UNet(input_dim, label_dim, hidden_dim)
# print(model)
# #print(model.contract1.parameters())
# 
# from torchviz import make_dot
# import matplotlib.pyplot as plt
# import graphviz
# 
# # dummy image input for plotting model
# x = torch.zeros(1, 4, 240, 240, dtype=torch.float, requires_grad=False)
# out = model(x)
# y = make_dot(out)
# # indicates the file name that will be written to
# y.render("RANO.gv") 
# =============================================================================