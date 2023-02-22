from nnunet.network_architecture.generic_UNet import Generic_UNet
from torch import nn
import torch

pool = 3

model = Generic_UNet(input_channels=3, 
                     base_num_features=64, 
                     num_classes=4, 
                     num_pool=pool)
# print("############################################")
# print(model)
# print("############################################")
# # child_counter = 0
# # for child in model.children():
# #     print(" child", child_counter, "is:")
# #     print(child)
# #     child_counter += 1
# print(list(model.conv_blocks_localization)[:])
# print("############################################")

new_model = nn.Sequential(*list(model.conv_blocks_localization)[:])
print(new_model)

# # remove the final linear layer of the regression model weights and bias
#         del checkpoint['state_dict']["Linear1.weight"]
#         del checkpoint['state_dict']["Linear1.bias"]

# load the existing model weights from the checkpoint
print("Model's state_dict:")
for param_tensor in new_model.state_dict():
    print(param_tensor, "\t", new_model.state_dict()[param_tensor].size())

model.load_state_dict(new_model.state_dict(), strict=False)

# freeze the weights if allow_update is false - leave unfrozen if allow_update is true
for param in model.conv_blocks_localization.parameters():
    param.requires_grad = True
    
print(new_model)

class joshNet(nn.Module):
    def __init__(self, new_model):
        super(joshNet, self).__init__()
        self.pretrained = new_model
        
        self.my_new_layers = nn.Sequential(nn.Linear((64*64*3*3), 8))

    def forward(self, x):
        x = self.pretrained(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layer(x)
        return x
    
# x = torch.zeros(1,1, 1,1, dtype=torch.float, requires_grad=False)
# model(x)

# prano = joshNet(new_model)
# prano(x)
                                           
# print(prano)
# print(new_model)
# # print(model)
# # input("jkdalslk")
# # print(model.items())

# # for this the model layers conv_blocks_localization[:][1] are the decoder and [0] are the encoder layers
# for i in range(pool):
#     print(model.conv_blocks_localization[i][1])
# #     model.conv_blocks_localization[i][1] = nn.Identify()
    


# for name, module in model.named_children():
#      print(module)

# print(model.layers)

# # model2 = nn.Sequential(*[model[i] for i in range(4)])

# for name, parameter in model.named_parameters():
#     if 'seg_outputs' in name:
#         print(f"parameter '{name}' will not be frozen")
#         parameter.requires_grad = True
#         parameter = nn.Identity()
#     else:
#         parameter.requires_grad = False
        
# # print(model)

# for name, parameter in model.named_parameters():
#     print(name)

# print(model2)
    
# model.named_parameters() = nn.Sequential(*[model.named_parameters[i] for i in range(len(model.named_parameters))[:-3]])
# print(model)

# model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
# print(model.classifier)