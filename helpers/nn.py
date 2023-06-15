import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
from helpers.utils import EarlyStopping


def np_to_torch(array, device):
    
    return torch.tensor(array.astype(np.float32)).to(device)
    
"""
NEURAL NET
"""

class NeuralNet(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNet, self).__init__()

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        #self.dropout1 = nn.Dropout2d(0.1)
        #self.dropout2 = nn.Dropout2d(0.1)

        # First fully connected layer
        self.fc1 = nn.Linear(input_shape, 64) # first size is output of flatten
        # Second fully connected layer that outputs our labels
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        #self.fc4 = nn.Linear(64, 1)

        
    # x represents our data
    def forward(self, x):

        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.dropout2(x)
        #x = self.fc4(x)
    
        # Apply softmax to x
        #output = F.log_softmax(x, dim=1)
        output = torch.sigmoid(x) # for BCE 
        
        return output
    
"""

# Architecture from CATHODE paper

class NeuralNet(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNet, self).__init__()

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        #self.dropout1 = nn.Dropout2d(0.1)
        #self.dropout2 = nn.Dropout2d(0.1)

        # First fully connected layer
        self.fc1 = nn.Linear(input_shape, 64) # first size is output of flatten
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        
    # x represents our data
    def forward(self, x):

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
    
        output = torch.sigmoid(x) # for BCE 
        
        return output
    

"""
    
