import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

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
        
        # input image = 240, conv1 = 120, conv2 = 60, conv3 = 30
        self.Linear1 = nn.Linear(((hidden_channels * 8) * 30 * 30), output_channels) 

    def forward(self, data_in):

        contract_0 = self.upfeature(data_in)
        
        contract_1 = self.contract1(contract_0)
        contract_2 = self.contract2(contract_1)
        contract_3 = self.contract3(contract_2)
        
        data_flat = torch.flatten(contract_3, start_dim=1)
        data_out = self.Linear1(data_flat)

        return data_out
    
# output testing for the model and saving of the model diagram usign graphiz
# 4 input MRI sequences, 1 segmentation output
# 16 hidden layers for this model
input_dim = 4
label_dim = 8
hidden_dim = 16

device = 'cuda'
                      
# to use the model with weights, frozen or unfrozen utilise UNet.load_weights
# to use the model without having any weights use UNet without allow_update and path
model = UNet(input_dim, label_dim, hidden_dim)
print(model)
#print(model.contract1.parameters())

from torchviz import make_dot
import matplotlib.pyplot as plt
import graphviz

# dummy image input for plotting model
x = torch.zeros(1, 4, 240, 240, dtype=torch.float, requires_grad=False)
out = model(x)
y = make_dot(out)
# indicates the file name that will be written to
y.render("RANO.gv") 