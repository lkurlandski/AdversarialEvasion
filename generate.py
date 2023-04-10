"""
AHHH so this isn't exactly random, but it selects the first 10 elements found but whatever....
"""


from __future__ import print_function
from collections import defaultdict
from pprint import pprint
import typing as tp
from typing import Generator, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn

from model import DLS_Model
from utils import get_highest_file


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad) -> Tensor:
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def show_ae_image(adv_examples, number, epsilons):
    fig = plt.figure(figsize=(8, 10))
    counter = 0
    for examples in range(len(adv_examples)):
        labels = [i[0] for i in adv_examples[examples]]
        output = [i[1] for i in adv_examples[examples]]
        images = [i[2] for i in adv_examples[examples]]
        for i in range(len(adv_examples[examples])):
            if labels[i] == number:
                plt.subplot(5, 5, counter + 1)
                counter += 1
                plt.tight_layout()
                plt.imshow(images[i], cmap="gray", interpolation="none")
                plt.title(
                    "Epsilon: {}\nOutput: {}".format(epsilons[examples], output[i])
                )
                plt.xticks([])
                plt.yticks([])
                break
    fig.savefig("ae_images.eps", format="eps")
    plt.close()


def generate_pgd(data, target, device, model, epsilon, alpha = 1e4, num_iter = 100, perturb_wrong = False):
        # Set requires_grad attribute of tensor. Important for Attack
        delta = torch.zeros_like(data, requires_grad=True)
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(data + delta), target)
            loss.backward()
            delta.data = (delta + data.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        perturbed_image = data + delta.detach()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image, True


def igsm_attack(model, image, label, epsilon, alpha = 1):
    model.eval()
    perturbed_image = image.clone().detach()
    num_iter = min(epsilon + 4, int(1.25 * epsilon))
    num_iter = 100  # FIXME: possibly remove
    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()

    return perturbed_image


def _adversarial_samples(
    model, data, target, attack, epsilon,
):
    data.requires_grad = True
    output = model(data)
    
    # I am lazy
    # If the initial prediction is wrong, dont bother attacking
    # init_pred = output.max(1, keepdim=True)[1]
    # if init_pred.item() != target.item():
    #     return adv_ex, target, final_pred
    
    alpha = 1e4
    alpha_igsm = 1
    num_iter = 100

    # Loop over all examples in test set
    if attack == "FGSM":
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

    elif attack == "PGD":
        # Set requires_grad attribute of tensor. Important for Attack
        delta = torch.zeros_like(data, requires_grad=True)
        for _ in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(data + delta), target)
            loss.backward()
            delta.data = (delta + data.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        perturbed_image = data + delta.detach()
        perturbed_data  = torch.clamp(perturbed_image, 0, 1)

    elif attack == "IGSM":
        perturbed_data = igsm_attack(model, data, target, epsilon, alpha_igsm)
        
    else:
        raise ValueError(f"{attack} not recongized.")

    final_pred = model(perturbed_data).max(1, keepdim=True)[1]
    return perturbed_data, target, final_pred


# Probably shouln't mess to much with this
def adversarial_samples(
    model, dataset, device, epsilon, attack, n_per_class=float("inf"), batch_size=1,
) -> Generator[Tuple[Tensor, Tensor, Tensor], None, bool]:
    """
    IMPORTANT:

    As long as the order of test_loader is random, simply selecting the first
    10 samples from each class is sufficient to select random examples.

    However, the order of the loader must be the same for each time this
    function is called, or else each call will end up with diffferent samples.
    Therefore, the loader's shuffle attribute must be false.

    Arguments
        model
        dataset
        Set n_per_class to float("inf") to generate adversarial samples for every
            training element
        Else set to an integer such as 10

    Yields
        Tuples of (adv_example, true labels, predicted labels)

    Returns
        Boolean of whether or not n_per_class samples were found for every class
    """
    
    # Figure out which indices to use
    tracker = defaultdict(lambda: 0)
    indices = []
    for i, (_, target) in enumerate(dataset):
        if tracker[target.item()] >= n_per_class:
            continue
        tracker[target.item()] += 1
        indices.append(i)
    
    # Create a batched dataloader that will use the selected indices
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.Subset(indices),
    )
    model.eval()
    
    # Loop through the specified indices and generate the examples
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        yield _adversarial_samples(model, data, target, attack, epsilon)

    return all(n == n_per_class for n in tracker.values())


# Not essential
# BROKEN
def accuracy_vs_epsilon(model, loader, epsilons, attack, n_per_class=10) -> tp.Dict[float, float]:

    accuracies = {}
    for epsilon in tqdm(epsilons, postfix="epsilon"):
        correct = 0
        itr = adversarial_samples(model, loader.dataset, DEVICE, epsilon, attack, n_per_class, batch_size=16)
        with tqdm(itr, total=n_per_class * 10, leave=False, postfix="sample") as pbar:
            for i, (adversarial_example, target, final_pred) in enumerate(itr, 1):
                model2(adversarial_example)
                if final_pred == target:
                    correct += 1
                pbar.update(1)
        print(f"{epsilon=}  {correct=}  {i=}")
        accuracies[epsilon] = correct / i

    return accuracies


# TODO
def test_model(model_1, dataset, epsilon, attack, model_2 = None) -> float:

    testing_model = model_2 if model_2 is not None else model_1
    correct = 0
    total = 0
    for adv_ex, target, _ in adversarial_samples(
        model_1, dataset, DEVICE, epsilon, attack, n_per_class=10, batch_size=16
    ):
        pred = testing_model(adv_ex)  # FIXME: get the label correctly
        if pred == target:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy
    
    



def task_2bc(attack):
    pretrained_model = get_highest_file("./models")

    # MNIST Test dataset and dataloader declaration
    # The sampler will cause the elements to be iterated in the same random order each epoch
    dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    # Initialize the network
    model = DLS_Model().to(DEVICE)
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=DEVICE))
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracy_vs_epsilon(model, loader, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], attack, f"figures/{attack}.png")


# Adjust this bit as nessecary
def main():

    task_2bc("FGSM")


if __name__ == "__main__":
    torch.manual_seed(0)
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
