from Net_modules.Loading_data import Load_Dataset
import Net_modules.Model_hyperparameters as Param
import Net.Unet_components_split as net

from Net_modules.Evaluation import Jaccard_Evaluation as Jacc
from Net_modules.Penalty import Penalty

from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import numpy as np
import logging
import shutil
import torch
import csv
import os
import random

seed = Param.Parameters.Network["Global"]["Seed"]

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = Param.Parameters.Network["Global"]["Enable_Determinism"]
torch.backends.cudnn.enabled = False

torch.set_num_threads(32)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(Param.Parameters.Network["Global"]["GPU"])

torch.cuda.empty_cache()

os.chdir(os.getcwd())

headers = ["Global","Hyperparameters","Train_paths"]
print("Torch version: ",torch.__version__)
print("")
for h in headers:
    for key, value in Param.Parameters.Network[h].items():
        print(f'{key: <30}{str(value): <35}')
        
np.set_printoptions(precision=4)
input("Press Enter to continue . . . ")

if os.path.exists("Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"]):
    print("Path already exists")
    print("Please enter Y to delete file contents or anything else to exit: ")
    replace = input("")
    if replace == "Y":
        shutil.rmtree("Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"])
        print("File deleted . . . continuing script")
    else:
        print("Exiting script")
        sys.exit()

np.set_printoptions(precision=4)

loss_f = Penalty(Param.Parameters.Network["Hyperparameters"]["Cosine_penalty"])
criterion = loss_f.MSELossorthogtest2_4

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def test(Test_data,mask_names):
    
    output_integer_for_printing = 0
    output_integer_for_printing_2 = 0

    np.random.seed(Param.Parameters.Network["Global"]["Seed"])
    torch.manual_seed(Param.Parameters.Network["Global"]["Seed"])
    Improvement = 0
    
    unet = net.UNet(Param.Parameters.Network["Hyperparameters"]["Input_dim"],
                    Param.Parameters.Network["Hyperparameters"]["Label_dim"], 
                    Param.Parameters.Network["Hyperparameters"]["Hidden_dim"]).to(
        Param.Parameters.Network["Global"]["device"])
    print(unet)

    if not os.path.exists("Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"] + "/Images"):
        os.makedirs("Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"] + "/Images")

    original = Param.Parameters.Network["Global"]["Param_location"]
    target = "Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"] + "Model_hyperparameters.py"

    shutil.copyfile(original, target)

    unet_opt = torch.optim.Adam(unet.parameters(), 
                                Param.Parameters.Network["Hyperparameters"]["Learning_rate"],
                                betas=Param.Parameters.Network["Hyperparameters"]["Betas"], 
                                weight_decay=Param.Parameters.Network["Hyperparameters"]["Weight_decay"])
    
    scaler = amp.GradScaler(enabled = True)

    cur_step = 0
    
    checkpoint = torch.load(Param.Parameters.Network["Test_paths"]["Load_path"])
    print(unet)
    unet.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']

    unet.eval()

    t, v, total_loss = [],[],[]

    running_loss, mse_run, cosine_run= 0.0, 0.0, 0.0
    loss_values,mse_values,cosine_values = [],[],[]
    jaccard,valid_jaccard = [],[]
    
    for truth_input, label_input in tqdm(Test_data):

        cur_batch_size = len(truth_input)

#         print("Truth", np.shape(truth_input), torch.sum(truth_input))#, np.sum(truth_input))
        truth_input = truth_input.to(Param.Parameters.Network["Global"]["device"])
        truth_input = truth_input.float() 
        truth_input = truth_input.squeeze()
#         for x_f in range(8):
#             print(torch.sum(truth_input[x_f]))
        
#         print("Truth", np.shape(label_input))#, np.sum(label_input))
        label_input = label_input.to(Param.Parameters.Network["Global"]["device"])
        label_input = label_input.float()
        label_input = label_input.squeeze()

        if cur_batch_size == 1:
            truth_input = truth_input[np.newaxis,:,:,:]
            label_input = label_input[np.newaxis,:]

        unet_opt.zero_grad()

        pred = unet(truth_input)

        label_input = np.squeeze(label_input)
        unet_loss, mse, cosine = criterion(pred, label_input)

        pred_output = pred.cpu().detach().numpy()
        truth_output = label_input.cpu().detach().numpy()
        
        pred_output = np.squeeze(pred_output)

#         print("truth", truth_output)
#         print("Pred", pred_output)

        for input_val in tqdm(range(cur_batch_size)):

            if Param.Parameters.Network["Hyperparameters"]["RANO"] == True:
                corners_truth, center_truth = Jacc.Obb(truth_output[input_val,:])
                mask_truth = Jacc.mask((240,240),corners_truth)*1
                corners_pred, center_pred = Jacc.Obb(pred_output[input_val,:])
                mask_pred = Jacc.mask((240,240),corners_pred)*1
            else:
                mask_truth = Jacc.BBox(truth_output[input_val,:])
                mask_pred = Jacc.BBox(pred_output[input_val,:])

            if np.sum(np.sum(mask_pred)) > 2:
                jac = jaccard_score(mask_truth.flatten(), mask_pred.flatten(), average='binary')
                jaccard.append(jac)
            else:
                jac = 0 #float("NaN")
                jaccard.append(0) #float("NaN"))
                
            # save here - jaccard and images
            
#             print("jac: ", jac)

            plt.imshow(np.zeros((240,240)))
            
            pred_output_show = pred_output[input_val,:]
            truth_output_show = truth_output[input_val,:]
            
            D1 = np.asarray([[pred_output_show[1],pred_output_show[3]],[pred_output_show[0],pred_output_show[2]]]) 
            D2 = np.asarray([[pred_output_show[5],pred_output_show[7]],[pred_output_show[4],pred_output_show[6]]]) 

            E1 = np.asarray([[truth_output_show[1],truth_output_show[3]],[truth_output_show[0],truth_output_show[2]]]) 
            E2 = np.asarray([[truth_output_show[5],truth_output_show[7]],[truth_output_show[4],truth_output_show[6]]]) 
            
#             print("image number = ", output_integer_for_printing)

            plt.plot(D1[0, :], D1[1, :], lw=2, c='y',label='_nolegend_')
            plt.plot(D2[0, :], D2[1, :], lw=2, c='y',label='Prediction')

            plt.plot(E1[0, :], E1[1, :], lw=2, c='b',label='_nolegend_')
            plt.plot(E2[0, :], E2[1, :], lw=2, c='b',label='Prediction')
            
            plt.axis('off')
            
#             plt.show()
            
            path_to_save_to = "Test_outputs/RANO/" + Param.Parameters.Network["Test_paths"]["Save_path"] + "/Images/" + mask_names[output_integer_for_printing]
            
            red = len(str(path_to_save_to.split("_")[-1]))
#             input("")
            
            if not os.path.exists(path_to_save_to[:-(red + 1)]):
                os.makedirs(path_to_save_to[:-(red + 1)])
                
                output_integer_for_printing_2 = 0
#                 print(path_to_save_to[:-(red + 1)] + "/image_output_" + str(output_integer_for_printing_2) + "_" + str(jac) + ".png")
#                 input("")
            
            plt.savefig(path_to_save_to[:-(red + 1)] + "/image_output_" + str(output_integer_for_printing_2) + "_" + str(jac) + ".png")
        
            plt.close() 
            
            output_integer_for_printing = output_integer_for_printing + 1
            output_integer_for_printing_2 = output_integer_for_printing_2 + 1
#             plt.show()
            
    print('Finished Testing Dataset')
    return 0

print("Loading Dataset")
folder = np.loadtxt(os.getcwd() + Param.Parameters.Network["Test_paths"]["Data_path"] + "/Training_dataset.csv", delimiter=",",dtype=str)

image_folder_in = folder[:,0]
masks_folder_in = folder[:,1]

dataset = Load_Dataset(os.getcwd() + Param.Parameters.Network["Test_paths"]["Data_path"],
                       image_folder_in,
                       masks_folder_in,
                       Param.Parameters.Network["Hyperparameters"]["Apply_Augmentation"])

Dataset_size = len(folder)

data_splits = [np.array([]), np.array([]), np.array([])]
data_split_values = [Param.Parameters.Network["Hyperparameters"]["Train_split"],
                     Param.Parameters.Network["Hyperparameters"]["Validation_split"],
                     Param.Parameters.Network["Hyperparameters"]["Test_split"]]

for splits in range(len(data_splits)):
    for val in data_split_values[splits]:
        if np.size(data_splits[splits]) == 0:
            data_splits[splits] = folder[np.where(folder[:,3].astype(int) == val),2]
        else:
            if np.size(folder[np.where(folder[:,3].astype(int) == val),2]) != 0:
                data_splits[splits] = np.concatenate((data_splits[splits],
                                                      folder[np.where(folder[:,3].astype(int) == val),2]), 
                                                      axis=1)

data_splits[0] = np.squeeze(data_splits[0])
data_splits[1] = np.squeeze(data_splits[1])
data_splits[2] = np.squeeze(data_splits[2])

test_data = data_splits[2].astype(int)

mask_names = folder[:,1]
print(data_splits[2][0])
print(data_splits[2][-1])
mask_names = mask_names[int(data_splits[2][0]):int(data_splits[2][-1])]

train_data = torch.utils.data.RandomSampler(data_splits[0],False)
validation_data = torch.utils.data.RandomSampler(data_splits[1],False)
del folder

Test_data=DataLoader(
    dataset=dataset,
    batch_size=Param.Parameters.Network["Hyperparameters"]["Batch_size"],
    num_workers=0,
    sampler=test_data, 
    pin_memory=True,
    worker_init_fn=_init_fn)

print("")
print("Actual test length", len(Test_data.sampler))

test(Test_data,mask_names)