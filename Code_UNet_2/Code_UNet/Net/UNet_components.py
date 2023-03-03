import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

import Net_modules.Parameters_SEG as Param

class Contract(nn.Module):
    def __init__(self, input_channels):
        super(Contract, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3,padding=1)
        self.batchnorm =  nn.BatchNorm2d(input_channels*2)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.maxpool(x)
        
        return x

class Expand(nn.Module):
    def __init__(self, input_channels):
        super(Expand, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3,padding=1)
        self.batchnorm =  nn.BatchNorm2d(input_channels//2)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.batchnorm(x)

        # https://github.com/xiaopeng-liao/Pytorch-UNet
        # /commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffY = skip_con_x.size()[2] - x.size()[2]
        diffX = skip_con_x.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_con_x, x], dim=1)
        x = self.conv2(x)
        x = self.batchnorm(x)
        
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        return x
    
class FeatureMap(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMap, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x

class Model(nn.Module):

    def __init__(self, input_channels, output_channels = 1, hidden_channels=32, Regress = False):
        super(Model, self).__init__()
        
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
        
        self.regress = Regress
        if Regress: output_channels = 8
            
        self.downfeature = FeatureMap(hidden_channels, output_channels)
        
        image_size = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"]

        # this is broken at the moment so we shall have to see how we get on.
        self.Linear1 = nn.Linear(output_channels, ((hidden_channels * 32) * 7 * 7)) 

    def forward(self, x):
        
        Debug = False
        
        x1 = self.upfeature(x)
        if Debug: print(x1.size())
        
        x2 = self.contract1(x1)
        if Debug: print(x2.size())
            
        x3 = self.contract2(x2)
        if Debug: print(x3.size())
            
        x4 = self.contract3(x3)
        if Debug: print(x4.size())
            
        x5 = self.contract4(x4)
        if Debug: print(x5.size())
            
        x6 = self.contract5(x5)
        if Debug: print(x6.size())
            
        if self.regress == True:
            
            x7 = torch.flatten(x6, start_dim=1)
            print(data_flat.size())
            
            data_out = self.Linear1(torch.mean(x7,0))
            print(data_out.size())
            
        if self.regress == False:
            
            x7 = self.expand1(x6, x5)
            if Debug: print(x7.size())

            x8 = self.expand2(x7, x4)
            if Debug: print(x8.size())
            
            x9 = self.expand3(x8, x3)
            if Debug: print(x9.size())
            
            x10 = self.expand4(x9, x2)
            if Debug: print(x10.size())
            
            x11 = self.expand5(x10, x1)
            if Debug: print(x11.size())

            data_out = self.downfeature(x11)
            if Debug: print(data_out.size())
                      
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