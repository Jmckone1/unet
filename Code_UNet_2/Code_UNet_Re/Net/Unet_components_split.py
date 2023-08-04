import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os

import Net_modules.Model_hyperparameters as Param

seed = Param.Parameters.Network["Global"]["Seed"]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = Param.Parameters.Network["Global"]["Enable_Determinism"]
torch.backends.cudnn.enabled = False

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
        torch.backends.cudnn.deterministic = Param.Parameters.Network["Global"]["Enable_Determinism"]
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class UNet(nn.Module):

    def __init__(self, input_channels, output_channels = 1, hidden_channels=32, Regress = False):
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
        
        self.regress = Param.Parameters.Network["Hyperparameters"]["Regress"]
        if Regress: output_channels = 8
            
        self.downfeature = FeatureMap(hidden_channels, output_channels)
        
        image_size = Param.Parameters.Network["Hyperparameters"]["Image_size"]

        if Param.Parameters.Network["Hyperparameters"]["Image_size"] == [240,240]:
            self.Linear_RANO = nn.Sequential(nn.Linear((hidden_channels*32)*7*7,8))
            self.Linear_BBox = nn.Sequential(nn.Linear((hidden_channels*32)*7*7,4))
        elif Param.Parameters.Network["Hyperparameters"]["Image_size"] == [256,256]:
            self.Linear_RANO = nn.Sequential(nn.Linear(int(512*(image_size[0]/32)*(image_size[1]/32)), 8))
            self.Linear_BBox = nn.Sequential(nn.Linear(int(512*(image_size[0]/32)*(image_size[1]/32)), 4))
        else:
            print("Not sure how you got here without the architecture defined but hey, regression wont work without this")
            import sys
            sys.exit()

    def forward(self, x):

        x1 = self.upfeature(x)
        
        x2 = self.contract1(x1)
        x3 = self.contract2(x2)
        x4 = self.contract3(x3)
        x5 = self.contract4(x4)
        x6 = self.contract5(x5)
        
        if self.regress == True:
#             print("Regress")
            if Param.Parameters.Network["Hyperparameters"]["RANO"] == True:
                x7 = torch.flatten(x6, start_dim=1)
                x8 = self.Linear_RANO(x7)
            elif Param.Parameters.Network["Hyperparameters"]["BBox"] == True:
                x7 = torch.flatten(x6, start_dim=1)
                x8 = self.Linear_BBox(x7)
            return x8
            
        if self.regress == False:
#             print("Segmentation")
            
            ex1 = self.expand1(x6, x5)
            
            ex2 = self.expand2(ex1, x4)
            ex3 = self.expand3(ex2, x3)
            ex4 = self.expand4(ex3, x2)
            ex5 = self.expand5(ex4, x1)
            
            data_out = self.downfeature(ex5)
#             x13 = torch.sigmoid(x12)
            
            return data_out
                              
    def load_weights(input_channels, output_channels = 1, hidden_channels = 32, Regress = True, Allow_update = False, Checkpoint_name = ""):
        # load the model and the checkpoint
        model = UNet(input_channels, output_channels, hidden_channels, Regress)
        checkpoint = torch.load(os.getcwd() + "/" + Checkpoint_name)
        
#         print(checkpoint['state_dict'])
        
        # if the regression is true here we will load the checkpoint and move on, no freezing will be done.
        if Regress == False:
            # remove the final linear layer of the regression model weights and bias
#             print(checkpoint['state_dict'])
#             input("")
            if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 1:
                del checkpoint['state_dict']["Linear.0.weight"]
                del checkpoint['state_dict']["Linear.0.bias"]
            if Param.Parameters.Network["Hyperparameters"]["Input_dim"] == 4:
                del checkpoint['state_dict']["Linear1.weight"]
                del checkpoint['state_dict']["Linear1.bias"]

            # load the existing model weights from the checkpoint
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            # freeze the weights if allow_update is false - leave unfrozen if allow_update is true
            for param in model.contract1.parameters():
                param.requires_grad = Allow_update
            for param in model.contract2.parameters():
                param.requires_grad = Allow_update
            for param in model.contract3.parameters():
                param.requires_grad = Allow_update
            for param in model.contract4.parameters():
                param.requires_grad = Allow_update
            for param in model.contract5.parameters():
                param.requires_grad = Allow_update
                      
        return model