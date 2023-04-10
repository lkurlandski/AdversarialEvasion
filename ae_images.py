# generate ae images
# mostly from the google drive link https://drive.google.com/drive/folders/1d43ZWPFDHcy8qo4fvMmmJI-B275cjE39 unless prefaced by commect #mod - which indicates paersonal modifcations
# use via 'python3 ae_attack.py [fgsm pgd igsm]'
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
import sys  # mod - sys argv
from model import DLS_Model
from utils import count_parameters, show_ae_image
from generate import pgd


def test_image_ae(
    model, test_loader, device, epsilon, attack
):  # mod- order changed for consistancy
    model.eval()
    correct = 0
    adv_examples = []
    number_class = [0] * 10  # mod - count for each class

    # Loop over all examples in test set
    for (
        data,
        target,
    ) in test_loader:  # mod - data has been shuffled when inserting, so it should be random already

        if number_class[int(target.item())] >= 10:
            continue

        if sum(number_class) >= 100:
            break

        number_class[int(target.item())] += 1

        # Send the data and label to the device
        if attack == "pgd":
            perturbed_data = pgd(data, target, device, model, epsilon)
        # Re-classify the perturbed image
        elif attack == "fgsm":
            data, target = data.to(device), target.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm(data, epsilon, data_grad)
        elif attack == "igsm":
            alpha_igsm = 1
            data, target = data.to(device), target.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            perturbed_data = igsm(model, data, target, epsilon)

        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples #mod - just save all the examples
            if epsilon == 0:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (target.item(), final_pred.item(), adv_ex)
                )  # mod - init_pred -> target: as guess incorrect from the start are not attacked, this isn't strictly needed but might as well
        else:
            # mod - just save all the examples

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((target.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / sum(number_class)
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, sum(number_class), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return epsilon, final_acc, adv_examples


# \\\\\\\\\\\\\\\\\\\\\\\ end of replicated code \\\\\\\\\\\\\\\\\\\\


use_cuda = True
print("Torch", torch.__version__, "CUDA", torch.version.cuda)
use_cuda = use_cuda and torch.cuda.is_available()

use_cuda = False  # mod - caveman multithreading

device = torch.device("cuda" if use_cuda else "cpu")
print(f"Model is using {device}")

attack_type = sys.argv[1]  # mod -sys argve attack type check
if (not attack_type == "fgsm") and (not attack_type == "pgd") and (not attack_type == "igsm"):
    print("invald attack type")

model = DLS_Model()
model = model.to(device)

BATCH_SIZE = 128
num_epoch = 2
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
attack_loader = (
    torch.utils.data.DataLoader(  # from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
    )
)


print(f"Model Parameters {count_parameters(model)/1000000}m")

print(model)
saved_model = "./mnist.pth"  # mod - change model path to test differently trained models
# saved_model = './mnist_pgd_pre.pth'
# if False:
if os.path.exists(saved_model):
    if use_cuda:
        model.load_state_dict(torch.load(saved_model))
    else:
        model.load_state_dict(torch.load(saved_model, map_location=torch.device("cpu")))
    print(f"Let's visualize some test samples")
    accuracies = []
    examples = []
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # mod - add epsilon
    # epsilons =  [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]  #mod - for training testing
    #
    # \\\\\\\\\\\\\\\\\\\\\\\ from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html \\\\\\\\\\\\\\\\\\\\
    # \\\\\\\\\\\\\\\\\\\\\\\ unless indicated by #mod which indicates personal modifications \\\\\\\\\\\\\\\\\\\\
    # Run test for each epsilon
    for eps in epsilons:
        print(eps)
        if attack_type == "fgsm":
            _, acc, ex = test_image_ae(
                model, attack_loader, device, eps, "fgsm"
            )  # mod - forlabel consistancy
        elif attack_type == "pgd":
            _, acc, ex = test_image_ae(
                model, attack_loader, device, eps, "pgd"
            )  # mod - forlabel consistancy
        elif attack_type == "igsm":
            _, acc, ex = test_image_ae(
                model, attack_loader, device, eps, "igsm"
            )  # mod - forlabel consistancy
        # _, acc, ex = test_fgsm_ae(model, attack_loader, device, eps)
        accuracies.append(acc)
        examples.append(ex)
    show_ae_image(examples, 5, epsilons)
# \\\\\\\\\\\\\\\\\\\\\\\ end of copied code \\\\\\\\\\\\\\\\\\\\
