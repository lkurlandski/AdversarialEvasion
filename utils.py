from __future__ import print_function
from pathlib import Path

import matplotlib.pyplot as plt
# import pandas as pd
from time import *


def get_models_path(attack: str, pretrained: bool, use_only_first_model: bool):
    return Path("models") / str(bool(pretrained)) / str(bool(use_only_first_model)) / str(attack)


def get_best_model_file(directory: str) -> Path:
    df = pd.read_csv(directory / "results.csv")
    i = df[["std_val_acc"]].idxmax()
    return directory / (df["epoch"][i] + ".pth")


def get_highest_file(directory: str) -> Path:
    directory = Path(directory)
    files = list(f for f in directory.iterdir() if f.stem.isdigit())
    if not files:
        raise FileNotFoundError((directory / "*").as_posix())

    ext = files[0].suffix
    latest = max((int(p.stem) for p in files))
    file = directory / (str(latest) + ext)
    return file


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_some_image(trainloader):
    examples = enumerate(trainloader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure(figsize=(8, 10))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
