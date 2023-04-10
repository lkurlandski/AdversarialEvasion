# AdversarialEvasion
CSEC-720: Deep Learning Security Project II

## Setup

```
conda create -name AdversarialEvasion -python=3.9
conda activate AdversarialEvasion
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tqdm==4.36.1 matplotlib==2.1.0 pandas
```

## Train and Evaluate

Use this script to train, harden, and evaluate classifiers. 

### Usage

Basic
```
python main.py
```

To query for options
```
python main.py --help
```

Example Usage With Flags
```
python main.py --attack=FGSM --pretrained=./models/False/False/None/3.pth --use_only_first_model
```

### Options

Experimental parameters
- --attack: Adversarial training technique, e.g., FGSM, IGSM, or PGD. To not perform any adversarial training, do not add the flag. Optional.
- --pretrained: .pth file of a pretrained model. If given, will finetune this model, presumably upon the adversarial examples. A good choice is ./models/False/False/None/3.pth, which is the highest performing model with no adversarial training. Optional. 
- --use_only_first_model: Flag, if supplied will only use the very first model to generate samples for training, i.e., all adversarial examples will be created from a whitebox attack upon the first model prior before any updates have been performed within the training loop

Configuration parameters
- --batch_size: Batch size for training and generating adversarial examples. Default 128
- --device: Hardware device, e.g. cpu, cuda:0, etc. Default cuda.
- --epochs: Number of epochs to train for. If a pretrained model exists that has been trained for this many epochs, will not perform training and will simply evaluate that model. Default 1.
- --seed: Seed to control random number generation. Default 0.

## Adversarial Generation

```
python generate.py
```

We can enhance this with a better CLI when the project comes together a little further.

## Output

-- models
   -- {PRETRAINED}
      -- {USE_ONLY_FIRST_MODEL}
	     -- {ATTACK}
			-- 1.pth
			-- ...
			-- N.pth
			-- report.csv
			-- best_at.csv
			-- best_std.csv


where 

- PRETRAINED is one of (True, False) and indicates whether or not the classifier was first trained on nonadversarial data then finetuned on adversarial and nonadversarial data for finetuning.
- USE_ONLY_FIRST_MODEL is one of (True, False) and indicates whether or not the classifier used to generate the adversarial examples is updated or if only the initial model (epoch 1 before training loop was entered) is used.
- ATTACK is one of (None, FGSM, IGSM, PGD) and refers to the attack that was used as part of the adversarial training.
- i.pth is a saved pytorch state dict
- report.csv contains the performances on validation data
- best_at.csv contains the test performance of the best performing model on adversarial validation data
- best_std.csv contains the test performance of the best performing model on nonadversarial validation data

