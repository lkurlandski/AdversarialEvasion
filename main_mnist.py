from __future__ import print_function
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from time import *
#%matplotlib inline

from model import DLS_Model
from utils import *

# specify the GPU device
use_cuda = True
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
print(f'Model is using {device}')


model = DLS_Model()
model = model.to(device)

BATCH_SIZE = 128
num_epoch = 2
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),])),batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),])),batch_size=BATCH_SIZE, shuffle=True)


print()
print(f'Model Parameters {count_parameters(model)/1000000}m')   

print(model)
saved_model = './mnist.pth'
if os.path.exists(saved_model):
    model.load_state_dict(torch.load(saved_model))
    print(f'Let\'s visualize some test samples')
    show_some_image(testloader)
    testing(model,testloader,device)
else:
    training(model, trainloader, device, optimizer, num_epoch, criterion)
    print(f'Let\'s visualize some test samples')
    show_some_image(testloader)
    testing(model,testloader,device)

