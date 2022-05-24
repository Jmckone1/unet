import torch
from torch import nn
import numpy as np

# i should save the outputs of both losses and combined losses for plotting - currently its just combined loss

class Penalty():
    def __init__(self, w1, w2):

        self.orth_weight = w1
        self.area_weight = w2
        self.a = nn.MSELoss()
        self.b = nn.CosineSimilarity()

    def MSELossorthog(self, output, target):
        batch = output.shape[0]
        mse = torch.ones(batch,device="cuda")
        
        loss = 0
        cosine = self.b(output,target)
        
        for i in range(output.shape[0]):
            mse[i] = self.a(output,target)
            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0

        cosine_mult = torch.abs(torch.mul(cosine, self.orth_weight))
        
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse), torch.mean(cosine_mult)