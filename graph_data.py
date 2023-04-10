from __future__ import print_function

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import sys
import typing as tp

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm
from time import *

from generate import _adversarial_samples
from model import DLS_Model
from utils import get_models_path, get_highest_file, show_some_image



def gen_ae(model_gen, test_loader, device, epsilon, attack): #generate ae, ten per class 
    adv_examples = []
    correct = []
    number_class = [0] * 10
    for data, target in testloader: #mod - data has been shuffled when inserting, so it should be random already
        if (number_class[int(target.item())] >= 10):
            continue
        
        if (sum(number_class) >= 100):
            break #when 100 samples
        
        number_class[int(target.item())] += 1
        data.to(device)
        perturbed_data, _, _ = _adversarial_samples(model_gen, data, target, attack, epsilon)
        
            
        adv_examples.append(perturbed_data)
        correct.append(target.item())
    return adv_examples, correct #list of adv examples and correct targets



if __name__ == "__main__":
    model_gen = sys.argv[1]
    model_test = sys.argv[2]
    attack = sys.argv[3]
    use_cuda = True
    print("Torch", torch.__version__, "CUDA", torch.version.cuda)
    use_cuda = use_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Model is using {device}")

    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    model = DLS_Model()
    model = model.to(device)

    model.load_state_dict(torch.load(model_gen))


    model.eval()



    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    #epsilons = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #mod - add epsilon
    epsilons =  [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]  #mod - for training testing

    adv_ex = []
    adv_ex_target = []
    for i in range(len(epsilons)): #generate and make list of lists of ae and targets
        temp_ex, temp_target = gen_ae(model, testloader, device, epsilons[i], attack)
        adv_ex.append(temp_ex)
        adv_ex_target.append(temp_target)

    model = DLS_Model()
    model = model.to(device)

    model.load_state_dict(torch.load(model_test))        
    correct_percentage = []
    for i in range(len(adv_ex)):#test target correct percetnage on test model
        correct = 0
        for j in range(len(adv_ex[i])):
            guess = model(adv_ex[i][j])
            if guess == adv_ex_target[i][j]:
                correct = correct+1
        correct_percentage.append(float (correct) / (float(len(adv_ex[i]))) )
    print(correct_percentage)
    