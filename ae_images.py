"""

"""

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from attack import generate_adversarial_samples
from model import DLS_Model
from utils import show_ae_image


attacks = ["FGSM", "IGSM", "PGD"]
saved_model = "/home/lk3591/Documents/courses/CSEC-720/AdversarialEvasion/output/False/False/None/18.pth"
clss = 5
epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

device = torch.device("cpu")
model = DLS_Model().to(device)
model.load_state_dict(torch.load(saved_model, map_location=device))
dataset = MNIST("./data", train=False, download=True, transform=Compose([ToTensor()]))

for attack in attacks:
    print(f"Let's visualize some test samples")
    examples = []
    for eps in tqdm(epsilons):
        examples.append([])
        gen = generate_adversarial_samples(
            model,
            dataset,
            device,
            eps,
            attack,
            n_per_class=10,
            batch_size=1,
        )
        for adv_ex, label, pred in tqdm(gen, total=100, leave=False):
            examples[-1].append((label.item(), pred.item(), adv_ex.squeeze().detach().cpu().numpy()))
    show_ae_image(examples, clss, epsilons, f"./figures/{clss}{attack}.png")
