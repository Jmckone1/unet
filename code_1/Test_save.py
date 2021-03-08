import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from code_1.Brats_dataloader_3 import BraTs_Dataset
import Net.Unet_components as net
import nibabel as nib
import os

torch.manual_seed(0)
np.random.seed(0)

# image interpolation multiplier
size = 1

# BCE with Logits loss, may change to soft dice
criterion = nn.BCEWithLogitsLoss()

n_epochs = 4
input_dim = 4
label_dim = 1
hidden_dim = 16

display_step = 100
batch_size = 16
lr = 0.0002
initial_shape = int(240 * size)
target_shape = int(240 * size)
device = 'cuda'

def dice_score(prediction, truth):
    # clip changes negative vals to 0 and those above 1 to 1
    pred_1 = np.clip(prediction, 0, 1.0)
    truth_1 = np.clip(truth, 0, 1.0)

    # binarize
    pred_1 = np.where(pred_1 > 0.5, 1, 0)
    truth_1 = np.where(truth_1 > 0.5, 1, 0)

    # Dice calculation
    product = np.dot(truth_1.flatten(), pred_1.flatten())
    dice_num = 2 * product + 1
    pred_sum = pred_1.sum()
    label_sum = truth_1.sum()
    dice_den = pred_sum + label_sum + 1
    score = dice_num / dice_den

    return score

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28),title=""):

    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.title(title)
    plt.imshow((image_grid.permute(1, 2, 0).squeeze()* 255).type(torch.uint8))
    plt.show()

def Test_save(Test_data, unet, unet_opt, save=False):
    unet.eval()
    img_num = 0
    pred_img = np.empty((240,240,155))
    for truth_input, label_input in tqdm(Test_data):

            cur_batch_size = len(truth_input)

            # flatten ground truth and label masks
            truth_input = truth_input.to(device)
            truth_input = truth_input.float() 
            truth_input = truth_input.squeeze()
            
            label_input = label_input.to(device)
            label_input = label_input.float()
            label_input = label_input.squeeze()
            
            # set accumilated gradients to 0 for param update
            unet_opt.zero_grad()
            pred = unet(truth_input)
            pred = pred.squeeze()
            
            show_tensor_images(truth_input[:,0,:,:], size=(label_dim, target_shape, target_shape),title="Real inputs")
            show_tensor_images(label_input, size=(label_dim, target_shape, target_shape),title="Real Labels")
            show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape),title="Predicted Output")

            plt.show()

            pred_output = pred.cpu().detach().numpy()
            truth_output = label_input.cpu().detach().numpy()
            DS = []
            for i in range(cur_batch_size):
                DS.append(dice_score(pred_output[i,:,:],truth_output[i,:,:]))
            print("Dice Score: ", DS)
            if save == True:
                for i in range(cur_batch_size):
                    pred_1 = np.clip(pred_output[i,:,:], 0, 1.0)
                    pred_1 = np.where(pred_1 > 0.5, 1, 0)
                    pred_img[:,:,img_num] = pred_1
                    
                    img_num = img_num + 1
    if save == True:
        pred_img_save = nib.Nifti1Image(pred_img, np.eye(4))
        nib.save(pred_img_save, os.path.join('predicted' + '.nii.gz'))  


#Train_loss,validation_loss = train(Train_data,Val_data)

unet = net.UNet(input_dim, label_dim, hidden_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=1e-8)

checkpoint = torch.load("Checkpoints/Checkpoints model_1/checkpoint_1.pth")

unet.load_state_dict(checkpoint['state_dict'])
unet_opt.load_state_dict(checkpoint['optimizer'])

#Test(Test_data_data, unet, unet_opt)
#Test(Val_data, unet, unet_opt)

dataset_single = BraTs_Dataset("Brats_2018 data", path_ext = ["/HGG_single"],size=size, apply_transform=False)
Single_data = DataLoader(
    dataset=dataset_single,
    batch_size=batch_size,
    shuffle=False)

Test_save(Single_data, unet, unet_opt, save=True)