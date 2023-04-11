"""
Adversarial example generation.

"""

from __future__ import print_function
from collections import defaultdict
import typing as tp
import torch
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy, nll_loss
from torch.utils.data import Dataset, DataLoader, Subset
from torch.types import Device

from model import DLS_Model


def fgsm(
    model: DLS_Model,
    data: Tensor,
    target: Tensor,
    epsilon: float,
) -> Tensor:
    """Fast Gradient Sign Method."""
    """https://pytorch.org/tutorials/beginner/fgsm_tutorial.html"""

    loss = nll_loss(model(data), target)
    model.zero_grad()
    loss.backward()
    sign_data_grad = data.grad.data.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def pgd(
    model: DLS_Model,
    data: Tensor,
    target: Tensor,
    epsilon: float,
    alpha: float = 1e4,
    num_iter: int = 100,
) -> Tensor:

    """Projected Gradient Descent."""
    """Tutorial that used for writing the PGD code: https://adversarial-ml-tutorial.org/adversarial_examples/"""

    delta = torch.zeros_like(data, requires_grad=True)
    for _ in range(num_iter):
        loss = CrossEntropyLoss()(model(data + delta), target)
        loss.backward()
        delta.data = (delta + data.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    perturbed_image = data + delta.detach()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def igsm(
    model: DLS_Model,
    image: Tensor,
    label: Tensor,
    epsilon: float,
    alpha: float = 1,
    num_iter: int = 100,
) -> Tensor:
    """Iterative gradient sign method."""
    """wrote the code by looking at the equation from the main paper (Kurakin et al.) https://arxiv.org/pdf/1607.02533.pdf"""

    model.eval()
    perturbed_image = image.clone().detach()
    num_iter = 100

    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()

    return perturbed_image



def adversarial_samples(
    model: DLS_Model, data: Tensor, target: Tensor, attack: str, epsilon: float
) -> tp.Tuple[Tensor, Tensor, Tensor]:
    """Create adversarial samples from given input data.

    Args:
        model (DLS_Model): model to attack
        data (Tensor): batch of input data to perturb
        target (Tensor): labels corresponding to the input data
        attack (str): attack to perform
        epsilon (float): constrains the size of the perturbation

    Raises:
        ValueError: if the attack is not recognized

    Returns:
        (Tensor): adversarial examples corresponding to the input data
        (Tensor): true labels of the adversarial data (same as input argument)
        (Tensor): predicted labels of the adversarial input data
    """
    data.requires_grad = True

    if attack == "FGSM":
        perturbed_data = fgsm(model, data, target, epsilon)
    elif attack == "PGD":
        perturbed_data = pgd(model, data, target, epsilon, 1e4, 100)
    elif attack == "IGSM":
        perturbed_data = igsm(model, data, target, epsilon, 1, 100)
    else:
        raise ValueError(f"{attack} not recongized.")

    final_pred = model(perturbed_data).max(1, keepdim=True)[1]
    return perturbed_data, target, final_pred


def generate_adversarial_samples(
    model: DataLoader,
    dataset: Dataset,
    device: Device,
    epsilon: float,
    attack: str,
    n_per_class: int = float("inf"),
    batch_size: int = 1,
) -> tp.Generator[tp.Tuple[Tensor, Tensor, Tensor], None, None]:
    """Iterative version of adversarial_samples that takes a full dataset.

    Args:
        model (DLS_Model): model to attack
        dataset (Dataset): data to select elements from
        device (Device): hardware device
        epsilon (float): constrains the size of the perturbation
        attack (str): attack to perform
        n_per_class (int, optional): Number of samples from each class to generate adversarial examples from.
            Defaults to float("inf"), in which case the entire dataset is used.
        batch_size (int, optional): Batch size for tensor computing. Defaults to 1.

    Yields:
        (Tensor): adversarial examples corresponding to the input data (batch_size=batch_size)
        (Tensor): true labels of the adversarial data (same as input argument)
        (Tensor): predicted labels of the adversarial input data

    Usage:
        >>> for perturbed_data, target, final_pred in generate_adversarial_samples(...):
        >>>     # perturbed_data is a Tensor
        >>>     # target is a Tensor
        >>>     # final_pred is a Tensor
        >>>     do_something(perturbed_data, target, final_pred)
    """
    
    model.eval()

    # Figure out which indices to use to include n_per_class in
    tracker = defaultdict(lambda: 0)
    indices = []
    for i, (_, target) in enumerate(dataset):
        if tracker[target] >= n_per_class:
            continue
        tracker[target] += 1
        indices.append(i)
    
    # Create a batched dataloader that will use the selected indices
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # sampler=Subset(dataset, indices),  # bug occurs don't know why
    )

    # Loop through the specified indices and generate the examples
    indices = set(indices)
    for i, (data, target) in enumerate(loader):
        if i not in indices:  # bug workaround
            continue
        data, target = data.to(device), target.to(device)
        yield adversarial_samples(model, data, target, attack, epsilon)
