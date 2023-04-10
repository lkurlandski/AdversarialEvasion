from __future__ import print_function
from pathlib import Path
import typing as tp

import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import Module
from torch.utils.data import DataLoader


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
                plt.title("Epsilon: {}\nOutput: {}".format(epsilons[examples], output[i]))
                plt.xticks([])
                plt.yticks([])
                break
    fig.savefig("ae_images.eps", format="eps")
    plt.close()


def get_output_path(
    pretrained: tp.Union[bool, str],
    use_only_first_model: tp.Union[bool, str],
    attack: tp.Optional[str],
) -> Path:
    return Path("output") / str(bool(pretrained)) / str(bool(use_only_first_model)) / str(attack)


def get_highest_file(directory: tp.Union[str, Path]) -> Path:
    directory = Path(directory)
    files = list(f for f in directory.iterdir() if f.stem.isdigit())
    if not files:
        raise FileNotFoundError((directory / "*").as_posix())

    ext = files[0].suffix
    latest = max((int(p.stem) for p in files))
    file = directory / (str(latest) + ext)
    return file


def get_best_models(directory: tp.Union[str, Path]) -> tp.Tuple[Path, Path]:
    report = directory / "report.csv"
    df = pd.read_csv(report)
    std_best = int(df[["std_val_acc"]].idxmax()) + 1
    at_best = int(df[["at_val_acc"]].idxmax()) + 1
    return directory / (str(std_best) + ".pth"), directory / (str(at_best) + ".pth")


def count_parameters(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_some_image(trainloader: DataLoader) -> None:
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
