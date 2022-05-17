import torch
from torch import nn
import numpy as np

class Penalty():
    def __init__(self, w1, w2):

        self.orth_weight = w1
        self.area_weight = w2
        self.loss = 0 # check if this needs to be in init
        self.a = nn.MSELoss()
        self.b = nn.CosineSimilarity()

    def MSELossorthog(self, output, target):
        batch = output.shape[0]
        mse = torch.ones(batch, dtype=torch.half,device="cuda")
        
        #self.loss = 0
        
        cosine = self.b(output,target)
        
        for i in range(output.shape[0]):
            mse[i] = self.a(output,target)
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            
        print("MSE",mse,mse.shape)
        print("COSINE",torch.mul(cosine, self.orth_weight), cosine.shape)
        
        self.loss = mse + torch.mul(cosine, self.orth_weight)

        return torch.mean(self.loss)