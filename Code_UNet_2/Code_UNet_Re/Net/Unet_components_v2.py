import Net_modules.Model_hyperparameters as Param

import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

torch.manual_seed(Param.Parameters.Network["Global"]["Seed"])

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

class Expand(nn.Module):
    def __init__(self, input_channels):
        super(Expand, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)

        # https://github.com/xiaopeng-liao/Pytorch-UNet
        # /commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffY = skip_con_x.size()[2] - x.size()[2]
        diffX = skip_con_x.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_con_x, x], dim=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        
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
                                  
        self.expand1 = Expand(hidden_channels * 32)
                              
        self.expand2 = Expand(hidden_channels * 16)
        self.expand3 = Expand(hidden_channels * 8)
        self.expand4 = Expand(hidden_channels * 4)
        self.expand5 = Expand(hidden_channels * 2)
        
        self.downfeature = FeatureMap(hidden_channels, output_channels)

    def forward(self, data_in):

        contract_0 = self.upfeature(data_in)
        
        contract_1 = self.contract1(contract_0)
        contract_2 = self.contract2(contract_1)
        contract_3 = self.contract3(contract_2)
        contract_4 = self.contract4(contract_3)
        contract_5 = self.contract5(contract_4)
        
        expand_1 = self.expand1(contract_5, contract_4)
        
        expand_2 = self.expand2(expand_1, contract_3)
        expand_3 = self.expand3(expand_2, contract_2)
        expand_4 = self.expand4(expand_3, contract_1)
        expand_5 = self.expand5(expand_4, contract_0)
        
        data_out = self.downfeature(expand_5)
                      
        return data_out
                              
    def load_weights(input_channels, output_channels, hidden_channels, model_name, allow_update = False):
        # load the model and the checkpoint
        model = UNet(input_channels, output_channels, hidden_channels)
        checkpoint = torch.load(model_name)

        # remove the final linear layer of the regression model weights and bias
        del checkpoint['state_dict']["Linear1.weight"]
        del checkpoint['state_dict']["Linear1.bias"]

        # load the existing model weights from the checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # freeze the weights if allow_update is false - leave unfrozen if allow_update is true
        for param in model.contract1.parameters():
            param.requires_grad = allow_update
        for param in model.contract2.parameters():
            param.requires_grad = allow_update
        for param in model.contract3.parameters():
            param.requires_grad = allow_update
        for param in model.contract4.parameters():
            param.requires_grad = allow_update
        for param in model.contract5.parameters():
            param.requires_grad = allow_update
                      
        return model
                       
# # output testing for the model and saving of the model diagram usign graphiz
# # 4 input MRI sequences, 1 segmentation output
# # 16 hidden layers for this model
# input_dim = 4
# label_dim = 1
# hidden_dim = 16

# device = 'cuda'
                      
# # to use the model with weights, frozen or unfrozen utilise UNet.load_weights
# # to use the model without having any weights use UNet without allow_update and path
# model = UNet.load_weights(input_dim, label_dim, hidden_dim,"Checkpoints_RANO/Unet_H16_M8/checkpoint_49.pth", allow_update=False)
# print(model)
# #print(model.contract1.parameters())

# from torchviz import make_dot
# import matplotlib.pyplot as plt
# import graphviz

# # dummy image input for plotting model
# x = torch.zeros(1, 4, 240, 240, dtype=torch.float, requires_grad=False)
# out = model(x)
# y = make_dot(out)
# # indicates the file name that will be written to
# y.render("frozen_2.gv") 