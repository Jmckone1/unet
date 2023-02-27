from Net_modules.SEG_dataloader import Load_Dataset
# from Net_modules.Penalty import Penalty
from Net_modules.Evaluation import DiceLoss
import Net_modules.Parameters_PRANO as Param
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import Unet_FULL_Train

torch.cuda.empty_cache()

os.chdir(os.getcwd())

headers = ["Global","Hyperparameters","Train_paths"]
print("Torch version: ",torch.__version__)
print("")
for h in headers:
    for key, value in Param.Parameters.PRANO_Net[h].items():
        print(f'{key: <30}{str(value): <35}')
        
("Press Enter to load Global Parameters . . . ")    
    
np.set_printoptions(precision=4)
print("Loaded seed")
np.random.seed(Param.Parameters.PRANO_Net["Global"]["Seed"])
torch.manual_seed(Param.Parameters.PRANO_Net["Global"]["Seed"])

print("Loaded GPU allocation")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(Param.Parameters.PRANO_Net["Global"]["GPU"])

print("Loaded cosine penalty")
loss_f = DiceLoss()#, Param.rNet.area_penalty)
criterion = loss_f

print("Loaded Dataset")
Full_Path = os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Data_path"]
print("")
dataset = Load_Dataset(Full_Path,
                       path_ext=Param.Parameters.PRANO_Net["Train_paths"]["Extensions"],
                       size=Param.Parameters.PRANO_Net["Hyperparameters"]["Image_size"],
                       apply_transform=False,
                       New_index=Param.Parameters.PRANO_Net["Hyperparameters"]["New_index"])

print("Loading index file")
index_f = np.load(os.getcwd() + Param.Parameters.PRANO_Net["Train_paths"]["Index_file"])
patients_number = len(index_f)

train_length = index_f[int(np.floor(patients_number*Param.Parameters.PRANO_Net["Hyperparameters"]["Train_split"]))]
validation_length = index_f[int(np.ceil(patients_number*Param.Parameters.PRANO_Net["Hyperparameters"]["Validation_split"]))]
test_length = index_f[int(np.ceil(patients_number*Param.Parameters.PRANO_Net["Hyperparameters"]["Test_split"]))]
all_data_length = index_f[-1]
custom_split = index_f[int(np.ceil(patients_number*Param.Parameters.PRANO_Net["Hyperparameters"]["Custom_split"]))]

train_range = list(range(0,train_length))
val_range = list(range(train_length,train_length+validation_length))
test_range = list(range(train_length+validation_length,train_length+validation_length+test_length))
all_data_range = list(range(0,all_data_length))
custom_split_range = list(range(0,custom_split))

# print(train_length)
# print(validation_length)
# print(test_length)
# print(all_data_length)
# print(custom_split)

train_data_m = torch.utils.data.RandomSampler(train_range,False)
validation_data_m = torch.utils.data.RandomSampler(val_range,False)
test_data_m = torch.utils.data.SubsetRandomSampler(test_range,False)
all_data_m = torch.utils.data.RandomSampler(all_data_range,False)
custom_split_m = torch.utils.data.RandomSampler(custom_split_range,False)

# https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67
print("Full_dataset: ", len(all_data_m))
print("Training: ", len(train_data_m))
print("validation: ", len(validation_data_m))

Train_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_size"],
    sampler=train_data_m)

Val_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.PRANO_Net["Hyperparameters"]["Batch_size"],
    sampler=validation_data_m)

# =============================================================================
# train(Train_data, Val_data, load=False)
# =============================================================================

train_model = Unet_FULL_Train.UNet_train(criterion)
train_model.train(Train_data, Val_data, load=False)