import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

# contracting path of the model
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

class UNet_contracting(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet_contracting, self).__init__()

        self.upfeature = FeatureMap(input_channels, hidden_channels)
        
        self.contract1 = Contract(hidden_channels)
        self.contract2 = Contract(hidden_channels * 2)
        self.contract3 = Contract(hidden_channels * 4)
        self.contract4 = Contract(hidden_channels * 8)
        self.contract5 = Contract(hidden_channels * 16)

    def forward(self, data_in):

        contract_0 = self.upfeature(data_in)
        
        contract_1 = self.contract1(contract_0)
        contract_2 = self.contract2(contract_1)
        contract_3 = self.contract3(contract_2)
        contract_4 = self.contract4(contract_3)
        contract_5 = self.contract5(contract_4)
        
        return contract_0,contract_1,contract_2,contract_3,contract_4,contract_5

#expanding path of the model
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
        # This is a fix for the skip conncetion dimesion padding to match with the other layers (check)
        diffY = skip_con_x.size()[2] - x.size()[2]
        diffX = skip_con_x.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_con_x, x], dim=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        
        return x

class UNet_expanding(nn.Module):

    def __init__(self, contract_layers, output_channels, hidden_channels=32):
        super(UNet_expanding, self).__init__()
        
        # the model here is a whole layer in additon to the prior unet segmentation example due to the RANO model responding much better with having the additonal layer, so it endeavours to match this, not sure if the problem with segmenation is here, have made a second file with a smaller model that will drop the layer.
        
        self.expand0 = Expand(hidden_channels * 32)
        self.expand1 = Expand(hidden_channels * 16)
        self.expand2 = Expand(hidden_channels * 8)
        self.expand3 = Expand(hidden_channels * 4)
        self.expand4 = Expand(hidden_channels * 2)
        
        self.downfeature = FeatureMap(hidden_channels, output_channels)

    def forward(self,contract_0,contract_1,contract_2,contract_3,contract_4,contract_5 ):

        expand_0 = self.expand0(contract_5,contract_4)
        expand_1 = self.expand1(expand_0,contract_3)
        expand_2 = self.expand2(expand_1,contract_2)
        expand_3 = self.expand3(expand_2,contract_1)
        expand_4 = self.expand4(expand_3,contract_0)
        
        data_out = self.downfeature(expand_4)
        
        return data_out
    
class Full_UNet(nn.Module):
    def __init__(self, contract, expand):
        super(Full_UNet, self).__init__()
        self.contract = contract
        self.expand = expand
        
    def forward(self,x1):
        contract_0,contract_1,contract_2,contract_3,contract_4,contract_5 = self.contract(x1)
        expand_layer = self.expand(contract_0,contract_1,contract_2,contract_3,contract_4,contract_5)
        # maybe there is something here that needs to happen, if i recall there was only one layer in the example that connected so i will have to further look into this
        #full_model = torch.cat((contract_0, expand_layer), dim=1)
        return expand_layer

def UNet(input_dim, label_dim, hidden_dim, model_name):
    # Create models and load state_dicts    
    Contracting_path = UNet_contracting(input_dim, label_dim, hidden_dim)
    Expanding_path = UNet_expanding(Contracting_path, label_dim, hidden_dim)

    # Load state dicts
    checkpoint = torch.load(model_name)
    
    # remove the final layer of the regression model weights and bias before merging the models
    del checkpoint['state_dict']["Linear1.weight"]
    del checkpoint['state_dict']["Linear1.bias"]

    Contracting_path.load_state_dict(checkpoint['state_dict'])

    #Freeze all layers in the pre-Trained model
    for param in Contracting_path.parameters():
        #param.requires_grad = False # Frozen
        param.requires_grad = True # Un-Frozen

    # concatenation of the contracting and expanding paths of the unet
    model = Full_UNet(Contracting_path, Expanding_path)

    return model

# input_dim = 4
# label_dim = 1
# hidden_dim = 16

# device = 'cuda'

# model = UNet(input_dim, label_dim, hidden_dim,"Checkpoints_RANO/Unet_H16_M8/checkpoint_49.pth")
# #print(model)
# from torchviz import make_dot
# import matplotlib.pyplot as plt
# import graphviz

# x = torch.zeros(1, 4, 240, 240, dtype=torch.float, requires_grad=False)
# out = model(x)

# print(model.named_parameters())
# y = make_dot(out)
# y.render("test_merged.gv")
# y
# print(y)