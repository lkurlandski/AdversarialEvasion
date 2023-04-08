from __future__ import print_function
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from time import *
#%matplotlib inline


def training(model, trainloader, device, optimizer, 
             num_epoch=2, criterion=nn.CrossEntropyLoss()):
    
    model.train()
    for epoch in range(1,num_epoch+1):
        with tqdm(trainloader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                inputs, labels = inputs.to(device), labels.to(device) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item())
                #sleep(0.0001)
    PATH = './mnist.pth'
    torch.save(model.state_dict(), PATH)

def testing(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100.0 * correct / total))

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_some_image(trainloader):
    examples = enumerate(trainloader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure(figsize=(8,10))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        
