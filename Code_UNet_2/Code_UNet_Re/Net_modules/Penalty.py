import Net_modules.Model_hyperparameters as Param
import torch
import numpy as np
import random

random.seed(Param.Parameters.Network["Global"]["Seed"])
np.random.seed(Param.Parameters.Network["Global"]["Seed"])
torch.manual_seed(Param.Parameters.Network["Global"]["Seed"])

class Penalty():
    def __init__(self, w1):

        self.orth_weight = w1
        self.a = torch.nn.MSELoss()
        self.b = torch.nn.CosineSimilarity()
        
    def Cosine_calc(self, target):
        
# #         print("COSINE", target.size())
#         if target.dim() == 1:
#             target = target[np.newaxis,:]

        x = target[:,0:4]
        y = target[:,4:8]

        cosine = self.b(x, y)

        return cosine
    
    def MSELossorthog(self, output, target):

        output = torch.squeeze(output)
        target = torch.squeeze(target)
        
# #         print("COSINE", target.size())
#         if target.dim() == 1:
#             target = target[np.newaxis,:]
        
        loss = 0
        batch = output.shape[0]
        # print(output.size(),target.size(),batch)
        
        mse = torch.ones(batch,device="cuda")
        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)
        
        
        for i in range(batch):
            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                stage_1 = cosine.data[i] # cosine output
                stage_2 = torch.sub(torch.abs(stage_1.data),0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                stage_3 = torch.mul(stage_2.data,2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
                stage_4 = torch.abs(stage_3.data) # set bounds to between 0 and 1
                cosine.data[i] = torch.mul(stage_4.data, self.orth_weight) # apply orthogonality weight
                
        cosine_mult = cosine.data.to(device='cuda') # torch.abs(torch.mul(cosine, self.orth_weight)).to(device='cuda')
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse), torch.mean(cosine_mult)