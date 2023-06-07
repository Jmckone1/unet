import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

class Penalty():
    def __init__(self, w1):

        self.orth_weight = w1
        self.a = nn.MSELoss()
        self.b = nn.CosineSimilarity()
        
    def Cosine_calc(self, target):

        print(target.size())    

        x = target[:,0:4]
        y = target[:,4:8]

        cosine = self.b(x, y)

        return cosine
    
    def MSELossorthog(self, output, target):
        
        batch = output.shape[0]
        print("Batch size", batch)
        print("Weighting Penalty", self.orth_weight)
        
        mse = torch.ones(batch,device="cuda")
        cosine = torch.ones(batch,device="cuda")

        loss = 0
        cosine = self.Cosine_calc(output)
        
        print("cosine_out", cosine)
        print("cosine_single", cosine[0])
        
        for i in range(output.shape[0]):
            mse[i] = self.a(output[i], target[i])
            print("Ground Truth", target[i])
            print("Prediction", output[i])

            if torch.isnan(cosine[i]) == True:
                cosine[i] = 0
            else:
                stage_1 = cosine.data[i] # cosine output
                stage_2 = torch.sub(torch.abs(stage_1.data),0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
                stage_3 = torch.mul(stage_2.data,2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
                stage_4 = torch.abs(stage_3.data) # set bounds to between 0 and 1
                cosine.data[i] = torch.mul(stage_4.data, self.orth_weight) # apply orthogonality weight
                
            print("cosine prediction:", cosine.data[i])
            print("_____________")
        cosine_mult = cosine.data.to(device='cuda') # torch.abs(torch.mul(cosine, self.orth_weight)).to(device='cuda')
        loss = torch.add(mse, cosine_mult)

        return torch.mean(loss), torch.mean(mse), torch.mean(cosine_mult)
    
if __name__ == "__main__":

    import Parameters as Param

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=Param.Global.GPU

    penalty = Penalty(1)

    truth = torch.ones(8,8)
    pred = torch.ones(8,8)
    
    #pred[0] = torch.tensor([84, 86, 86, 94, 92, 95, 92, 96])
    pred[0] = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1])
    pred[1] = torch.tensor([46,48,46,72,52,52,41,60])
    pred[2] = torch.tensor([0, 1, 1, 0, -1, 0, 1, 1.1])
    #pred[2] = torch.tensor([55, 54, 87, 87, 70, 68, 59, 69])
    pred[3] = torch.tensor([55, 44, 87, 87, 55, 54, 97, 87])
    pred[4] = torch.tensor([55, 44, 87, 87, 55, 44, 87, 87])
    pred[5] = torch.tensor([55, 44, 87, 87, 55, 34, 87, 87])
    pred[6] = torch.tensor([95, 44, 87, 87, 68, 98, 97, 37])

    print(pred)

    x,y,z = penalty.MSELossorthog(pred,truth)
    
    print("output", z)
    
    for i in range(len(pred)):
        x_maj = [pred[i][1],pred[i][3]]
        x_min = [pred[i][0],pred[i][2]]
        y_maj = [pred[i][5],pred[i][7]]
        y_min = [pred[i][4],pred[i][6]]
        
        plt.plot(x_maj,y_min)
        plt.plot(y_maj,x_min)
        plt.show()
        
        print("what?", z)
        print("input", pred[i])
        #print("output",z[i])

#         stage_1 = z[i] # cosine output
#         stage_2 = torch.sub(stage_1,0.5) # subtract 0.5 to set orthog from 0.5 to 0 and parallel from 0 and 1 to 0.5 and -0.5
#         stage_3 = torch.mul(stage_2,2) # set bounds to between -1 and 1 - this step isnt necessarily needed but looks nicer
#         stage_4 = torch.abs(stage_3) # set bounds to between 0 and 1
#         stage_5 = torch.mul(stage_4,orth_W) # apply orthogonality weight
        
#         print("output_split -- ", stage_5)