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
        
        self.expand1 = Expand(hidden_channels * 16)
        
        self.expand2 = Expand(hidden_channels * 8)
        self.expand3 = Expand(hidden_channels * 4)
        self.expand4 = Expand(hidden_channels * 2)
        
        self.downfeature = FeatureMap(hidden_channels, output_channels)

    def forward(self, data_in):

        contract_0 = self.upfeature(data_in)
        
        contract_1 = self.contract1(contract_0)
        contract_2 = self.contract2(contract_1)
        contract_3 = self.contract3(contract_2)
        contract_4 = self.contract4(contract_3)
        
        expand_5 = self.expand1(contract_4, contract_3)
        
        expand_6 = self.expand2(expand_5, contract_2)
        expand_7 = self.expand3(expand_6, contract_1)
        expand_8 = self.expand4(expand_7, contract_0)
        
        data_out = self.downfeature(expand_8)
        
        return data_out