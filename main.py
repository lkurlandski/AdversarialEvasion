"""
Train and evaluate the classifiers.
"""

from __future__ import print_function

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import typing as tp

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.nn.functional import cross_entropy
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.types import Device
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from attack import adversarial_samples
from model import DLS_Model
from utils import get_best_models, get_highest_file, get_output_path


TESTING_EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
TRAINING_EPSILONS = [0.05, 0.1, 0.2, 0.25, 0.3]
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1


def train(
    model: Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device: Device,
    optimizer: Optimizer,
    output_path: Path,
    start_epoch: int = 1,
    num_epoch: int = 2,
    criterion: Module = cross_entropy,
    standard: bool = True,
    attack: str = None,
    epsilons: list = None,
    use_only_first_model: bool = False,
) -> None:
    """Train the model and evaluate upon a validation set."""

    def report(epoch: int, tr_loss: float, std_val_acc: float, at_val_acc: float) -> None:
        """Append to the report file."""
        with open(output_path / "report.csv", "a") as handle:
            handle.write(f"{epoch},{tr_loss},{std_val_acc},{at_val_acc}\n")

    def update(X: Tensor, y: Tensor) -> Tensor:
        """Update the weights of the model."""
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        return loss

    # Initialize the results file
    with open(output_path / "report.csv", "w") as handle:
        handle.write("epoch,tr_loss,std_val_acc,at_val_acc\n")

    with tqdm(range(start_epoch, num_epoch + 1), unit="batch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model.train()
            tr_loss = 0
            for inputs, labels in tqdm(trainloader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                # Control which model will be used to generate the adversarial examples
                if not use_only_first_model:
                    adversarial_source_model = model
                elif epoch == start_epoch:
                    adversarial_source_model = deepcopy(model)
                    adversarial_source_model.load_state_dict(model.state_dict())
                    adversarial_source_model.eval()
                # Update model with the standard training data
                if standard:
                    loss = update(inputs, labels)
                    tr_loss += loss.item()
                # Update model with adversarial examples
                if attack is not None:
                    for epsilon in epsilons:
                        adv_exs, targets, _, = adversarial_samples(
                            adversarial_source_model, inputs, labels, attack, epsilon
                        )
                        update(adv_exs.to(device), targets.to(device))
                
            # Get performance on the validation sets
            std_val_acc, at_val_acc = evaluate(model, valloader, device, standard, attack, epsilons)
            # End of iteration tasks
            report(epoch, tr_loss, std_val_acc, at_val_acc)
            tepoch.set_postfix(acc=f"std_val_acc={round(std_val_acc,1)} | at_val_acc={round(at_val_acc,1)}")
            torch.save(model.state_dict(), output_path / f"{epoch}.pth")


def evaluate(
    model: Module,
    testloader: DataLoader,
    device: Device,
    standard: bool = True,
    attack: str = None,
    epsilons: tp.List[float] = None,
) -> tp.Tuple[float, float]:
    """Evaluate the model upon a test or validation set."""

    model.eval()
    # Tracks the performance on standard examples and adversarial examples
    std_correct, std_total = 0, 0
    at_correct, at_total = 0, 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        # Evaluate model with the standard training data
        if standard:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            std_total += labels.size(0)
            std_correct += (predicted == labels).sum().item()
        # Evaluate model with adversarial examples
        if attack is not None:
            for epsilon in epsilons:
                adv_exs = adversarial_samples(model, images, labels, attack, epsilon)[0]
                outputs = model(adv_exs)
                _, predicted = torch.max(outputs.data, 1)
                at_total += labels.size(0)
                at_correct += (predicted == labels).sum().item()

    std = (100.0 * std_correct / std_total) if standard is not None else -1
    at = (100.0 * at_correct / at_total) if attack is not None else -1
    return std, at


def main() -> None:
    """Run the training, validation, and evaluation."""

    torch.manual_seed(SEED)
    output_path = get_output_path(PRETRAINED, USE_ONLY_FIRST_MODEL, ATTACK)

    # Get the datasets and dataloaders
    dataset = MNIST("./data", train=True, download=True, transform=Compose([ToTensor()]))
    sizes = [int(TRAIN_SPLIT * len(dataset)), int(VAL_SPLIT * len(dataset))]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, sizes)
    test_dataset = MNIST("./data", train=False, download=True, transform=Compose([ToTensor()]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = DLS_Model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Otherwise, find the most recently trained model or the pretrained model and load it
    saved_model = PRETRAINED
    latest_epoch = 0
    if saved_model is None:
        try:
            saved_model = get_highest_file(output_path)
            latest_epoch = int(saved_model.stem)
        except FileNotFoundError:
            pass
    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))

    # If the model still needs to be trained, train it
    output_path.mkdir(parents=True, exist_ok=True)
    if latest_epoch < EPOCHS:
        print("TRAINING...")
        train(
            model,
            trainloader,
            valloader,
            DEVICE,
            optimizer,
            output_path,
            latest_epoch + 1,
            EPOCHS,
            criterion,
            standard=True,
            attack=ATTACK,
            epsilons=TRAINING_EPSILONS,
            use_only_first_model=USE_ONLY_FIRST_MODEL,
        )

    print("EVALUATING...")
    # Now evaluate the best performing models based on their validation set accuracies
    std_best, at_best = get_best_models(output_path)
    for o, i in [("best_std.csv", std_best), ("best_at.csv", at_best)]:
        model = DLS_Model().to(DEVICE)
        model.load_state_dict(torch.load(i))
        std_acc, at_acc = evaluate(
            model, testloader, DEVICE, standard=True, attack=ATTACK, epsilons=TESTING_EPSILONS
        )
        epoch = i.stem
        with open(output_path / o, "w") as handle:
            handle.write("epoch,std_acc,at_acc\n")
            handle.write(f"{epoch},{std_acc},{at_acc}")
            
    print("COMPLETE")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Optional adversarial training technique, e.g., `FGSM`, `IGSM`, or `PGD`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training and generating adversarial examples.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Hardware device, e.g. `cpu`, `cuda:0`, etc."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run. If a pretrained model exists that has been trained for this many epochs, will not perform training.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model. If given, will finetune this model, presumably upon the adversarial examples.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to control random number generators."
    )
    parser.add_argument(
        "--use_only_first_model",
        action="store_true",
        default=False,
        help="If True, will only use the very first model to generate samples for training.",
    )
    args = parser.parse_args()

    ATTACK = args.attack
    BATCH_SIZE = args.batch_size
    DEVICE = torch.device(args.device)
    EPOCHS = args.epochs
    PRETRAINED = Path(args.pretrained) if args.pretrained is not None else None
    SEED = args.seed
    USE_ONLY_FIRST_MODEL = args.use_only_first_model

    main()
