# import torch
# from torch import nn
# import numpy as np

# # i should save the outputs of both losses and combined losses for plotting - currently its just combined loss

# class Penalty():
#     def __init__(self, w1, w2):

#         self.orth_weight = w1
#         self.area_weight = w2
#         self.a = nn.MSELoss()
#         self.b = nn.CosineSimilarity()

#     def MSELossorthog(self, output, target):
#         batch = output.shape[0]
#         mse = torch.ones(batch,device="cuda")
        
#         loss = 0
#         cosine = self.b(output,target)
        
#         for i in range(output.shape[0]):
#             mse[i] = self.a(output,target)
#             if torch.isnan(cosine[i]) == True:
#                 cosine[i] = 0

#         cosine_mult = torch.abs(torch.mul(cosine, self.orth_weight))
        
#         loss = torch.add(mse, cosine_mult)

#         return torch.mean(loss), torch.mean(mse), torch.mean(cosine_mult)


import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
import os


    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# i should save the outputs of both losses and combined losses for plotting - currently its just combined loss

class Penalty():
    def __init__(self, w1, w2):

        self.orth_weight = w1
        self.area_weight = w2
        self.a = nn.MSELoss()
        self.b = nn.CosineSimilarity()
        
    def Cosine_calc(self, target):
        
#         print(target)
#         input("")
        
        x_maj = [target[1],target[3]]
        x_min = [target[0],target[2]]
        y_maj = [target[5],target[7]]
        y_min = [target[4],target[6]]
        
#         print(target.shape)
        x = target[:,0:4]
        y = target[:,4:8]
        
#         print(x)
#         print(y)
#         input("")
        
#         m1 = torch.tensor([target[0:4]])
#         m2 = torch.tensor([target[4:7]])
#         print(m1,m2)
        
# #         m3 = torch.tensor([target[0],target[1],target[2],target[3]])
# #         m4 = torch.tensor([target[4],target[5],target[6],target[7]])
        
#         print(m1,m2)
        cosine = self.b(x, y)
        print(cosine)
        print("_____________")
        
        return cosine

    def MSELossorthog(self, output, target):
        
        batch = output.shape[0]
        print("Batch size", batch)
        mse = torch.ones(batch,device="cuda")
        cosine = torch.ones(batch,device="cuda")
        
        loss = 0
        cosine = self.Cosine_calc(output)
        
        for i in range(output.shape[0]):
            mse[i] = self.a(output[i], target[i])
            print(output[i])
            
            
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0

        cosine_mult = torch.abs(torch.mul(cosine, self.orth_weight))
        
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse), torch.mean(cosine_mult)
    
if __name__ == "__main__":
    
    penalty = Penalty(1,1)

    truth = torch.ones(8,8)
    pred = torch.ones(8,8)
    
    pred[4] = torch.tensor([55, 54, 87, 87, 70, 68, 59, 69])
    pred[0] = torch.tensor([84, 86, 86, 94, 92, 95, 92, 96])
    pred[1] = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1])
    pred[2] = torch.tensor([55, 54, 87, 87, 70, 68, 59, 69])
    pred[3] = torch.tensor([55, 44, 87, 87, 55, 54, 97, 87])
    #pred[4] = torch.tensor([55, 54, 87, 87, 70, 68, 59, 69])
    pred[5] = torch.tensor([1, 1, 0, 0, 0, 1, 0, 1])
    pred[6] = torch.tensor([55, 54, 87, 87, 70, 68, 59, 69])
    pred[7] = torch.tensor([55, 54, 87, 87, 70, 68, 89, 69])

    print(pred)
    penalty.MSELossorthog(pred,truth)