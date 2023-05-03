import torch
import torch.nn as nn
from model import model
user_embedding = torch.randn(7, 8)
item_embedding = torch.randn(6, 8)
model = model(8)
user_feature = model(user_embedding[1, :], item_embedding, False, False)
item_feature = model(item_embedding[0, :], user_embedding, False, True)
predicted2 = torch.matmul(user_feature, item_embedding[2, :])
predicted1 = torch.matmul(user_embedding[0, :], item_feature)
loss = predicted1+ predicted2
model.zero_grad()
loss.backward()
print("ada")

