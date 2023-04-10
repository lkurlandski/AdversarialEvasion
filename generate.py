"""
Generate stuff.

TODO:
- Nothing in here works at the moment
"""


from __future__ import print_function
import typing as tp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from attack import generate_adversarial_samples
from model import DLS_Model


def accuracy_vs_epsilon(model, loader, epsilons, attack, n_per_class=10) -> tp.Dict[float, float]:

    accuracies = {}
    for epsilon in tqdm(epsilons, postfix="epsilon"):
        correct = 0
        itr = generate_adversarial_samples(
            model, loader.dataset, DEVICE, epsilon, attack, n_per_class, batch_size=16
        )
        with tqdm(itr, total=n_per_class * 10, leave=False, postfix="sample") as pbar:
            for i, (adversarial_example, target, final_pred) in enumerate(itr, 1):
                model2(adversarial_example)
                if final_pred == target:
                    correct += 1
                pbar.update(1)
        print(f"{epsilon=}  {correct=}  {i=}")
        accuracies[epsilon] = correct / i

    return accuracies


def test_model(model_1, dataset, epsilon, attack, model_2=None) -> float:

    testing_model = model_2 if model_2 is not None else model_1
    correct = 0
    total = 0
    for adv_ex, target, _ in generate_adversarial_samples(
        model_1, dataset, DEVICE, epsilon, attack, n_per_class=10, batch_size=16
    ):
        pred = testing_model(adv_ex)  # FIXME: get the label correctly
        if pred == target:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def task_2bc(attack):
    pretrained_model = "./model.pth"

    # MNIST Test dataset and dataloader declaration
    # The sampler will cause the elements to be iterated in the same random order each epoch
    dataset = MNIST("./data", train=False, download=True, transform=Compose([ToTensor()]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Initialize the network
    model = DLS_Model().to(DEVICE)
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=DEVICE))
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracy_vs_epsilon(
        model,
        loader,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        attack,
        f"figures/{attack}.png",
    )


# Adjust this bit as nessecary
def main():

    task_2bc("FGSM")


if __name__ == "__main__":
    torch.manual_seed(0)
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
