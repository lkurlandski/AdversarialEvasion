# AdversarialEvasion
CSEC-720: Deep Learning Security Project II

## Setup

```
conda create -name AdversarialEvasion -python=3.9
conda activate AdversarialEvasion
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tqdm==4.36.1 matplotlib==2.1.0
```

## Train Classifier

```
python main_mnist.py
```

Feel free to cancel once loss no longer decreases or whatever. Checkpointing system will reload the latest trained model from the ./models directory. However, validation accuracy plataeus after around 5 epochs..

## 

