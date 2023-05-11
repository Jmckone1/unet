# import torch
# from torch import nn
# from tqdm.auto import tqdm
# import torch.nn.functional as F

# class Contract(nn.Module):
#     def __init__(self, input_channels):
#         super(Contract, self).__init__()
        
#         self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3,padding=1)
#         self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3,padding=1)
#         self.activation = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = self.maxpool(x)
        
#         return x
    
# class FeatureMap(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(FeatureMap, self).__init__()
        
#         self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

#     def forward(self, x):
        
#         x = self.conv(x)
        
#         return x

# class UNet(nn.Module):

#     def __init__(self, input_channels, output_channels, hidden_channels=32):
#         super(UNet, self).__init__()

#         self.upfeature = FeatureMap(input_channels, hidden_channels)
        
#         self.contract1 = Contract(hidden_channels)
#         self.contract2 = Contract(hidden_channels * 2)
#         self.contract3 = Contract(hidden_channels * 4)
        
#         # input image = 240, conv1 = 120, conv2 = 60, conv3 = 30
#         self.Linear1 = nn.Linear(((hidden_channels * 8) * 30 * 30), output_channels) 

#     def forward(self, data_in):

#         contract_0 = self.upfeature(data_in)
        
#         contract_1 = self.contract1(contract_0)
#         contract_2 = self.contract2(contract_1)
#         contract_3 = self.contract3(contract_2)
        
#         data_flat = torch.flatten(contract_3, start_dim=1)
#         data_out = self.Linear1(data_flat)

#         return data_out
    
    
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np

class Contract(nn.Module):
    def __init__(self, input_channels):
        super(Contract, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.drop = nn.Dropout(0.4)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        #x = self.drop(x)
        
        return x
    
class FeatureMap(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMap, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class UNet(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()

        self.upfeature = FeatureMap(input_channels, hidden_channels)
        
        self.contract1 = Contract(hidden_channels)
        self.contract2 = Contract(hidden_channels * 2)
        self.contract3 = Contract(hidden_channels * 4)
        self.contract4 = Contract(hidden_channels * 8)
        self.contract5 = Contract(hidden_channels * 16)
        
        # input image = 240, conv1 = 120, conv2 = 60, conv3 = 30 15 7 4
        self.Linear1 = nn.Linear(((hidden_channels * 32) * 7 * 7), output_channels) # if size = 1
#         self.Linear1 = nn.Linear(((hidden_channels * 32) * 3 * 3), output_channels) # if size = 0.5
#         self.Linear1 = nn.Linear(((hidden_channels * 32) * 1 * 1), output_channels) # if size = 0.25

    def forward(self, data_in):
        
        
        if(data_in.ndim == 3):
            data_in = data_in[:,np.newaxis,:,:]
            
#         print(data_in.shape)
        contract_0 = self.upfeature(data_in)
#         print(contract_0.shape)
        contract_1 = self.contract1(contract_0)
#         print(contract_1.shape)
        contract_2 = self.contract2(contract_1)
#         print(contract_2.shape)
        contract_3 = self.contract3(contract_2)
#         print(contract_3.shape)
        contract_4 = self.contract4(contract_3)
#         print(contract_4.shape)
        contract_5 = self.contract5(contract_4)
#         print(contract_5.shape)
        
        data_flat = torch.flatten(contract_5, start_dim=1)
#         print(data_flat.shape)
        data_out = self.Linear1(data_flat)

        return data_out