import torch.nn as nn
import torch.nn.functional as F
import torch
from Net.pytorch_dcsaunet.encoder import CSA
import numpy as np

import Net_modules.Parameters_SEG as Param

csa_block = CSA()

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(1, channels, kernel_size, padding=  kernel_size // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        
        # this is in the case of a single image being left at the end of the training set so that the single channel is retained.
        if x.ndim==4:
            x = torch.squeeze(x)
            x = x[:,np.newaxis,:,:]
            
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    

# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=4, n_classes=1, Regress = False):
        super(Model, self).__init__()
        self.pfc = PFC(64)
        self.img_channels = img_channels
        self.n_classes = n_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        self.up_conv1 = Up()
        self.up_conv2 = Up()
        self.up_conv3 = Up()
        self.up_conv4 = Up()
        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4
        self.up1 = csa_block.layer5
        self.up2 = csa_block.layer6
        self.up3 = csa_block.layer7
        self.up4 = csa_block.layer8
        
        self.regress = Regress
        
#         # make a note somewhere on how exactly this works and what the numbers mean because
#         # i keep forgetting a feel really dumb . . .
        image_size = Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"]
        # print(image_size)
        self.Linear = nn.Sequential(nn.Linear(int(512*(image_size[0]/16)*(image_size[1]/16)), 8))
       
    def forward(self, x):
        
        Debug = True
  
        x1 = self.pfc(x)
        if Debug: print(x1.size())
        x2 = self.maxpool(x1)
        if Debug: print(x2.size())
        
        x3 = self.down1(x2)   
        if Debug: print(x3.size())
        x4 = self.maxpool(x3)
        if Debug: print(x4.size())
        
        x5 = self.down2(x4)
        if Debug: print(x5.size())
        x6 = self.maxpool(x5)
        if Debug: print(x6.size())
        
        x7 = self.down3(x6)
        if Debug: print(x6.size())
        x8 = self.maxpool(x7)
        if Debug: print(x8.size())
        
        x9 = self.down4(x8)
        if Debug: print(x9.size())
        
        if self.regress == True:
            
            x10 = torch.flatten(x9, start_dim=1)
            if Debug: print(x10.size())
            x11 = self.Linear(x10)
            if Debug: print(x11.size())
            return x11
            
        if self.regress == False:

            x10 = self.up_conv1(x9,x7)
            if Debug: print(x10.size())
            x11 = self.up1(x10)
            if Debug: print(x11.size())

            x12 = self.up_conv2(x11,x5) 
            if Debug: print(x12.size())
            x13 = self.up2(x12)
            if Debug: print(x13.size())

            x14 = self.up_conv3(x13,x3)  
            if Debug: print(x14.size())
            x15 = self.up3(x14)
            if Debug: print(x15.size())

            x16 = self.up_conv4(x15,x1)
            if Debug: print(x16.size())
            x17 = self.up4(x16)
            if Debug: print(x17.size())

            x18 = self.out_conv(x17)
            if Debug: print(x18.size())

            #x19 = torch.sigmoid(x18)
            return x18
        
    def load_weights(img_channels=4, n_classes=1, Regress = True, Allow_update = False, Checkpoint_name = ""):
        # load the model and the checkpoint
        model = Model(img_channels, n_classes, Regress)
        checkpoint = torch.load(Checkpoint_name)
        
#         print(checkpoint['state_dict'])
        input("Press to continue . . . ")
        # remove the final linear layer of the regression model weights and bias
        del checkpoint['state_dict']["Linear.0.weight"]
        del checkpoint['state_dict']["Linear.0.bias"]

        # load the existing model weights from the checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if Regress == False:
            # freeze the weights if allow_update is false - leave unfrozen if allow_update is true
            for param in model.pfc.parameters():
                param.requires_grad = Allow_update
            for param in model.down1.parameters():
                param.requires_grad = Allow_update
            for param in model.down2.parameters():
                param.requires_grad = Allow_update
            for param in model.down3.parameters():
                param.requires_grad = Allow_update
            for param in model.down4.parameters():
                param.requires_grad = Allow_update

    #         x1 = self.pfc(x)
    #         x2 = self.maxpool(x1)
    #         x3 = self.down1(x2)   
    #         x4 = self.maxpool(x3)
    #         x5 = self.down2(x4)
    #         x6 = self.maxpool(x5)
    #         x7 = self.down3(x6)
    #         x8 = self.maxpool(x7)
    #         x9 = self.down4(x8)
                      
        return model