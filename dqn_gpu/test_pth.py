import torch

file_path = "trained_DQN.pth"
model = torch.load(file_path)
print(model)