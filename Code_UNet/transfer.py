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

    def forward(self, data_in):

        contract_0 = self.upfeature(data_in)
        
        contract_1 = self.contract1(contract_0)
        contract_2 = self.contract2(contract_1)
        contract_3 = self.contract3(contract_2)
        contract_4 = self.contract4(contract_3)
        contract_5 = self.contract5(contract_4)

        return contract_1,contract_2,contract_3,contract_4,contract_5

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
        diffY = skip_con_x.size()[2] - x.size()[2]
        diffX = skip_con_x.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_con_x, x], dim=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        
        return x

class UNet_2(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()

        self.expand0 = Expand(hidden_channels * 32)
        self.expand1 = Expand(hidden_channels * 16)
        self.expand2 = Expand(hidden_channels * 8)
        self.expand3 = Expand(hidden_channels * 4)
        self.expand4 = Expand(hidden_channels * 2)
        
        self.downfeature = FeatureMap(hidden_channels, output_channels)
        
        # input image = 240, conv1 = 120, conv2 = 60, conv3 = 30 15 7 4
        #self.Linear1 = nn.Linear(((hidden_channels * 32) * 7 * 7), output_channels) 

    def forward(self, data_in):
        
        expand_1 = self.expand0(contract_5,contract_4)
        expand_2 = self.expand1(expand_1,contract_3)
        expand_3 = self.expand2(expand_2,contract_2)
        expand_4 = self.expand3(expand_3,contract_1)
        expand_5 = self.expand4(expand_4,contract_0)
        
        data_out = self.downfeature(expand_5)
        
        return data_out

class MyMerge(nn.Module):
    def __init__(self, contract, expand):
        super(MyEnsemble, self).__init__()
        self.contract = contract
        self.expand = expand
        
    def forward(self, x):
        x1 = self.contract(x)
        x2 = self.expand(x1)
        return x2

# Create models and load state_dicts    
modelA = UNet()
modelB = UNet_2()
# Load state dicts
modelA.load_state_dict(torch.load(PATH))

model = MyEnsemble(modelA, modelB)
x = torch.randn(1, 10)
output = model(x)