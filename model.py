from __future__ import print_function
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DLS_Model(nn.Module):
    def __init__(self):
        super(DLS_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.pool1((self.conv1(x))))
        x = F.relu(self.pool2((self.conv2(x))))
        x = F.relu(self.pool3((self.conv3(x))))

        #print(x.shape)
        x = x.view(-1, 128 * 3 * 3)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
