from model.attention_module import Attention_Module
import torch
model = Attention_Module(3, 6)
print(model)
x = torch.randn([1,3,224, 224])
print(model(x).shape)