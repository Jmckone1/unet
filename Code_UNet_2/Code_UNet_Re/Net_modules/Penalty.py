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

        if target.dim() == 1:
            target = target[np.newaxis,:]

        x = target[:,0:4]
        y = target[:,4:8]

        cosine = self.b(x, y)

        return cosine
    
    def MSELossorthog(self, output, target):

#         output = torch.squeeze(output)
#         target = torch.squeeze(target)

#         if target.dim() == 1:
#             target = target[np.newaxis,:]
        
        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")
        if Param.Parameters.Network["Hyperparameters"]["RANO"] == True:
            cosine = torch.ones(batch,device="cuda")
            cosine = self.Cosine_calc(output)

        if Param.Parameters.Network["Hyperparameters"]["RANO"] == True:
            for i in range(batch):

                mse[i] = self.a(output[i,:], target[i,:])
                if torch.isnan(cosine[i]) == True:
                    cosine[i] = 0
                else:
                    stage_1 = cosine.data[i] # cosine output
                    stage_2 = torch.sub(torch.abs(stage_1.data), 0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                    stage_3 = torch.mul(stage_2.data, 2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
                    stage_4 = torch.abs(stage_3.data) # set bounds to between 0 and 1
                    cosine.data[i] = torch.mul(stage_4.data, self.orth_weight) # apply orthogonality weight

            cosine_mult = cosine.data.to(device='cuda') # torch.abs(torch.mul(cosine, self.orth_weight)).to(device='cuda')
            loss = torch.add(mse, cosine_mult)
            
            return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine_mult).detach()
        else:
            for i in range(batch):
                mse[i] = self.a(output[i,:], target[i,:])
            loss = mse
            
            return torch.mean(loss), torch.mean(mse).detach(), torch.mean(mse).detach()
    
    def MSELossorthogtest1(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        for i in range(batch):
            mse[i] = self.a(output[i,:], target[i,:])
            loss = mse
            
        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(mse).detach()
    
    def MSELossorthogtest2(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine.data[i] = 0

        cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
    
    def MSELossorthogtest2_1(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                cosine[i] = torch.mul(cosine[i], self.orth_weight)

#         cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
    
    def MSELossorthogtest2_2(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
#                 print("Orth abs")
                cosine[i] = torch.abs(cosine[i].clone())
#                 print("Orth sub")
                cosine[i] = torch.sub(cosine[i].clone(), 0.5)
#                 print("Orth weight")
                cosine[i] = torch.mul(cosine[i].clone(), self.orth_weight)
                
#         cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine)
        
        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
    
    def MSELossorthogtest2_3(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                cosine[i] = torch.abs(cosine[i].clone())
                cosine[i] = torch.sub(cosine[i].clone(), 0.5)
                cosine[i] = torch.mul(cosine[i].clone(), 2)
                cosine[i] = torch.abs(cosine[i].clone())
                cosine[i] = torch.mul(cosine[i].clone(), self.orth_weight)

        loss = torch.add(mse, cosine)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
    
    def MSELossorthogtest2_4(self,output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")
        if Param.Parameters.Network["Hyperparameters"]["RANO"] == True:

            cosine = torch.ones(batch,device="cuda")
            cosine = self.Cosine_calc(output)

            for i in range(batch):

                mse[i] = self.a(output[i,:], target[i,:])
                if torch.isnan(cosine[i]) == True:
                    cosine[i] = 0
                else:
                    cosine[i] = torch.abs(cosine[i].clone())
                    cosine[i] = torch.mul(cosine[i].clone(), self.orth_weight)

            loss = torch.add(mse, cosine)
            return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
        else:
            for i in range(batch):
                mse[i] = self.a(output[i,:], target[i,:])

            loss = mse
            return torch.mean(loss), torch.mean(mse).detach(), torch.mean(mse).detach()
                
    def MSELossorthogtest3(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                cosine[i] = torch.mul(cosine[i], self.orth_weight)
                 # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                
        cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine_mult).detach()
    
    def MSELossorthogtest4(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                cosine[i] = torch.sub(torch.abs(cosine[i]), 0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                cosine[i] = torch.mul(cosine[i], 2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
                cosine[i] = torch.abs(cosine[i]) # set bounds to between 0 and 1
                cosine[i] = torch.mul(cosine[i], self.orth_weight)
                
#         cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine).detach()
        
    def MSELossorthogtest5(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")

        cosine = torch.ones(batch,device="cuda")
        cosine = self.Cosine_calc(output)

        for i in range(batch):

            mse[i] = self.a(output[i,:], target[i,:])
            
            if torch.isnan(cosine.data[i]) == True:
                cosine.data[i] = 0
            else:
                cosine.data[i] = torch.sub(torch.abs(cosine.data[i]), 0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                cosine.data[i] = torch.mul(cosine.data[i], 2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
                cosine.data[i] = torch.abs(cosine.data[i]) # set bounds to between 0 and 1
                cosine.data[i] = torch.mul(cosine.data[i], self.orth_weight)
                
        cosine_mult = cosine.data.to(device='cuda')
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse).detach(), torch.mean(cosine_mult).detach()
    
    def MSELossorthogtest6(self, output, target):

        loss = 0
        batch = output.shape[0]
        
        mse = torch.ones(batch,device="cuda")
        cosine = torch.ones(batch,device="cuda")

        for i in range(batch):
            mse[i] = self.a(output[i,:], target[i,:])
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                cosine[i] = self.Cosine_calc(output[i,:])
                print(cosine[i])
                stage_1 = cosine.data[i] # cosine output
                print(stage_1)
                print(stage_1.data)
                cosine.data[i] = torch.sub(torch.abs(stage_1.data),0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                print(cosine.data[i])
                print(cosine[i])
#                 stage_3 = torch.mul(stage_2.data,2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
#                 stage_4 = torch.abs(stage_3.data) # set bounds to between 0 and 1
#                 cosine.data[i] = torch.mul(stage_4.data, self.orth_weight) # apply orthogonality weight
                
        cosine_mult = cosine.data.to(device='cuda') # torch.abs(torch.mul(cosine, self.orth_weight)).to(device='cuda')
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss.detach()), torch.mean(mse.detach()), torch.mean(cosine_mult.detach())