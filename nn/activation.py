import torch

def linear(x):
    return x

def relu(x):
    return torch.relu(x)

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def gaussian(x):
    return torch.exp(-torch.square(x))

def cauchy(x):
    return 1.0 / (1.0 + torch.square(x))
