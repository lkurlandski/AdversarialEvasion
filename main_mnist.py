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

torch.manual_seed(0)

# specify the GPU device
use_cuda = True
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
print(f'Model is using {device}')


model = DLS_Model()
model = model.to(device)

BATCH_SIZE = 128
num_epoch = 5
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(.9 * len(dataset)), int(.1 * len(dataset))])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


print()
print(f'Model Parameters {count_parameters(model)/1000000}m')   

print(model)

# Added a checkpointing system
try:
    saved_model = get_highest_file("./models")
    latest_epoch = int(saved_model.stem)
except FileNotFoundError:
    saved_model = None
    latest_epoch = 0

if saved_model is not None:
    model.load_state_dict(torch.load(saved_model))

if latest_epoch < num_epoch:
    training(model, trainloader, valloader, device, optimizer, latest_epoch+1, num_epoch, criterion)

print(f'Let\'s visualize some test samples')
show_some_image(testloader)
print("Test accuracy: ", testing(model,testloader,device))
