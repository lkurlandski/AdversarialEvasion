from __future__ import print_function

from argparse import ArgumentParser
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


def training(
    model,
    trainloader,
    valloader,
    device,
    optimizer,
    start_epoch=1,
    num_epoch=2,
    criterion=CrossEntropyLoss(),
    standard: bool = True,
    attack: str = None,
    epsilons: list = None,
):
    """Training.

    Args:
        model
        trainloader
        valloader
        device
        optimizer
        start_epoch
        num_epoch
        criterion
        standard: if True, includes standard data into loss
        attack: the adversarial attack to use
        epsilons: the epsilons for the attack
        **kwargs: keyword args for the attack
    """
    
    def report(epoch, tr_loss, std_val_acc, at_val_acc):
        with open(report_file, "a") as handle:
            handle.write(f"{epoch},{tr_loss},{std_val_acc},{at_val_acc}\n")
    

    def update(X, y) -> Tensor:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        return loss

    path = get_models_path(attack)
    path.mkdir(exist_ok=True)
    report_file = path / f"report.csv"
    with open(report_file, "w") as handle:
        handle.write(f"epoch,tr_loss,std_val_acc,at_val_acc\n")

    with tqdm(range(start_epoch, num_epoch + 1), unit="batch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model.train()
            tr_loss = 0
            for inputs, labels in tqdm(trainloader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                if standard:
                    loss = update(inputs, labels)
                    tr_loss += loss.item()

                if attack is not None:
                    for epsilon in epsilons:
                        adv_exs, targets, _, = _adversarial_samples(model, inputs, labels, attack, epsilon)
                        update(adv_exs.to(device), targets.to(device))

            std_val_acc, at_val_acc = testing(model, valloader, device, standard, attack, epsilons)
            report(epoch, tr_loss, std_val_acc, at_val_acc)
            tepoch.set_postfix(acc=f"{std_val_acc=} | {at_val_acc=}")
            torch.save(model.state_dict(), path / f"{epoch}.pth")


def testing(
    model, testloader, device, standard: bool = True, attack: str = None, epsilons: tp.List[float] = None,
) -> tp.Tuple[float, float]:
    model.eval()
    std_correct, std_total = 0, 0
    at_correct, at_total = 0, 0
    
    # Need grad for the adversarial examples...
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        if standard:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            std_total += labels.size(0)
            std_correct += (predicted == labels).sum().item()
        
        if attack is not None:
            for epsilon in epsilons:
                adv_exs, _, _, = _adversarial_samples(model, images, labels, attack, epsilon)
                outputs = model(adv_exs)
                _, predicted = torch.max(outputs.data, 1)
                at_total += labels.size(0)
                at_correct += (predicted == labels).sum().item()

    return (100.0 * std_correct / std_total), (100.0 * at_correct / at_total)


def main():
    torch.manual_seed(SEED)

    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))]
    )
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
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DLS_Model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    try:
        saved_model = get_highest_file(get_models_path(ATTACK))
        latest_epoch = int(saved_model.stem)
    except FileNotFoundError:
        saved_model = None
        latest_epoch = 0

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))

    if latest_epoch < EPOCHS:
        training(
            model,
            trainloader,
            valloader,
            DEVICE,
            optimizer,
            latest_epoch + 1,
            EPOCHS,
            criterion,
            standard=True,
            attack=ATTACK,
            epsilons=[0.05, 0.1, 0.2, 0.25, 0.3],
        )

    print(f"Let's visualize some test samples")
    show_some_image(testloader)
    print("Test accuracy: ", testing(model, testloader, DEVICE))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ATTACK = args.attack
    BATCH_SIZE = args.batch_size
    DEVICE = torch.device(args.device)
    EPOCHS = args.epochs
    SEED = args.seed

    main()
