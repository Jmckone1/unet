import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from skimage import io
import numpy as np
from Brats_dataloader import BraTs_Dataset
from torch.autograd import Variable
#from eval import eval_net

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28),title=""):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.title(title)
    plt.imshow((image_grid.permute(1, 2, 0).squeeze()* 255).type(torch.uint8))
    plt.show()

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    shape_x1 = image.shape[2]
    shape_x2 = image.shape[2] - new_shape[2]
    shape_y1 = image.shape[3]
    shape_y2 = image.shape[3] - new_shape[3]
    
    cropped_image = image[0:image.shape[0],
                          0:image.shape[1],
                          math.ceil(shape_x2/2):shape_x1 - math.floor(shape_x2/2),
                          math.ceil(shape_y2/2):shape_y1 - math.floor(shape_y2/2)]
    return cropped_image

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3,padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        #print(x.shape,skip_con_x.shape)
        x = self.upsample(x)
        x = self.conv1(x)

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffY = skip_con_x.size()[2] - x.size()[2]
        diffX = skip_con_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_con_x, x], dim=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x
    
class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a UNet - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)
        return xn

criterion = nn.BCEWithLogitsLoss()
n_epochs = 2
input_dim = 4
label_dim = 1
display_step = 5
batch_size = 16
lr = 0.0002
initial_shape = 240
target_shape = 240
device = 'cuda'
val_percent = 0.1
# https://nipy.org/nibabel/gettingstarted.html

plt.gray()

def Validate(model: nn.Module, criterion, Val_data):
    print("Validation...")
    model.eval()
    loss = []
    for real, labels in tqdm(Val_data):
        real = real.to(device)
        real = real.float() 
        real = real.squeeze()
        labels = labels.to(device)
        labels = labels.float()
        labels = labels.squeeze()

        pred = model(real)
        loss = criterion(pred, labels)
        losses.append(loss.data[0])

    loss = np.mean(losses)  # type: float

    print('Valid loss: {:.5f}'.format(valid_loss))
    metrics = loss
    return metrics

def train():
    # 0 whole tumour region
    # https://www.med.upenn.edu/sbia/brats2018/data.html
    # 1 = non-enhancing tumor core (necrotic region)
    # 2 = peritumoral edema (Edema)
    # 4 = GD-enhancing tumor (Active)
    # any other undefined input uses no blank labels - doesnt function
    dataset=BraTs_Dataset("HGG",label_val=0)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    Train_data = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True)

    Val_data = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)
    
    # used in validation
    # unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=2)
    cur_step = 0

    total_loss = []
    epoch_loss = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        loss_values = []
        for real, labels in tqdm(Train_data):

            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            real = real.float() 
            real = real.squeeze()
            labels = labels.to(device)
            labels = labels.float()
            labels = labels.squeeze()

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            pred = pred.squeeze()
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            running_loss =+ unet_loss.item() * real.size(0)
            print('Training loss: {:.5f}'.format(running_loss))

            if cur_step > 0:
              if cur_step % display_step == 0:
                  print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                  #plt.subplot(1, 3, 1)
                  show_tensor_images(
                      crop(real, torch.Size([len(real), 1, target_shape, target_shape]))[:,1,:,:], 
                      size=(1, target_shape, target_shape),
                      title="Flair Input Channel ( channel 2 of 4 )"
                  )
                  #plt.subplot(1, 3, 2)
                  show_tensor_images(labels, size=(label_dim, target_shape, target_shape),title="Real Labels")
                  #plt.subplot(1, 3, 3)
                  show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape),title="Predicted Output")
                  plt.plot(range(len(loss_values)),loss_values)
                  plt.show()
            cur_step += 1
            loss_values.append(running_loss / len(Train_data))
            
        plt.plot(range(len(loss_values)),loss_values)
        plt.title("Epoch " + str(epoch + 1) + ": loss")

        plt.show()
        total_loss.append(loss_values)
        epoch_loss.append(unet_loss.item())

        # need to add functions/ add additional loops for the validation stage
        # val_score = eval_net(unet, Val_data, device)
        valid_loss = Validate(unet, criterion, Val_data)
        # scheduler.step(val_score)
        
    print('Finished Training Trainset')
    return total_loss, valid_loss


Train_loss,validation_loss = train()

# NEXT STEPS: sort out a multi-class classification for the labels - bitwise or of labelled regions
# comment the rest of the code and lpit it up into folders, sort it out in general
# sort out validation and testing splits
# get evaluation metrics. maybe dice score? accuracy?
# look into checkpoints saving and loading to reduce the time requirement each time
# get this code onto github