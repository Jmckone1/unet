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

#         #self.Linear = nn.Sequential(nn.Linear(((hidden_channels * 32) * 7 * 7), 8))
#         print("values", 256*(image_size[0]/16*7)*(image_size[1]/16*7))
#         print("values2", int(256*(image_size[0]/14)*(image_size[1]/14)))
#         print("values3", (hidden_channels * 32) * 7 * 7)
        
        # Due to the differences in architecture (one defining the hidden layers (UNet) and the other not (DGSAU)) 
        # there requires a different shape for the linear layers which is why this section exists.
        if Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"] == [240,240]:
            self.Linear = nn.Sequential(nn.Linear((hidden_channels*32)*7*7,8))
        elif Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"] == [256,256]:
            self.Linear = nn.Sequential(nn.Linear(int(256*(image_size[0]/16)*(image_size[1]/16)), 8))
        else:
            print("Not sure how you got here without the architecture defined but hey, regression wont work without this")
            import sys
            sys.exit()

    def forward(self, x):
        
        Debug = Param.Parameters.PRANO_Net["Global"]["Debug"]
        
        x1 = self.upfeature(x)
        if Debug: 
            print("1", x1.size())
            print("x1 - upfeature", torch.cuda.memory_allocated()/1024**2)
        
        x2 = self.contract1(x1)
        if Debug: 
            print("2", x2.size())
            print("x2 - contract1", torch.cuda.memory_allocated()/1024**2)
            
        x3 = self.contract2(x2)
        if Debug: 
            print("3", x3.size())
            print("x3 - contract2", torch.cuda.memory_allocated()/1024**2)
            
        x4 = self.contract3(x3)
        if Debug: 
            print("4", x4.size())
            print("x4 - contract3", torch.cuda.memory_allocated()/1024**2)
            
        x5 = self.contract4(x4)
        if Debug: 
            print("5", x5.size())
            print("x5 - contract4", torch.cuda.memory_allocated()/1024**2)
            
        x6 = self.contract5(x5)
        if Debug: 
            print("6", x6.size())
            print("x6 - contract5", torch.cuda.memory_allocated()/1024**2)
            
        if self.regress == True:
            
            x7 = torch.flatten(x6, start_dim=1)
            if Debug: 
                print("7", x7.size())
                print("x7 - Flatten", torch.cuda.memory_allocated()/1024**2)
            
            x8 = self.Linear(x7)
            if Debug:
                print("8", x8.size())
                print("x8 - Linear", torch.cuda.memory_allocated()/1024**2)
            
            return x8
            
        if self.regress == False:
            
            x7 = self.expand1(x6, x5)
            if Debug: 
                print("7", x7.size())
                print("x7 - expand1", torch.cuda.memory_allocated()/1024**2)

            x8 = self.expand2(x7, x4)
            if Debug: 
                print("8", x8.size())
                print("x8 - expand2", torch.cuda.memory_allocated()/1024**2)
            
            x9 = self.expand3(x8, x3)
            if Debug: 
                print("9", x9.size())
                print("x9 - expand3", torch.cuda.memory_allocated()/1024**2)
            
            x10 = self.expand4(x9, x2)
            if Debug: 
                print("10", x10.size())
                print("x10 - expand4", torch.cuda.memory_allocated()/1024**2)
            
            x11 = self.expand5(x10, x1)
            if Debug: 
                print("11", x11.size())
                print("x11 - expand5", torch.cuda.memory_allocated()/1024**2)

            x12 = self.downfeature(x11)
            if Debug: 
                print("12", x12.size())
                print("x12 - downfeature", torch.cuda.memory_allocated()/1024**2)
            
            x13 = torch.sigmoid(x12)
            
            return x13
                              
    def load_weights(input_channels, output_channels = 1, hidden_channels = 32, Regress = True, Allow_update = False, Checkpoint_name = ""):
        
        # load the model and the checkpoint
        model = Model(input_channels, output_channels, hidden_channels, Regress)
        checkpoint = torch.load(Checkpoint_name)
        
        # if the regression is true here we will load the checkpoint and move on, no freezing will be done.
        if Regress == False:
            # remove the final linear layer of the regression model weights and bias
            del checkpoint['state_dict']["Linear.0.weight"]
            del checkpoint['state_dict']["Linear.0.bias"]

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