# General Imports
import numpy as np
import os
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt

# Torch imports
import torch
import torch.nn as nn 
import torch.optim as optim

# Functions
from modules.train import train_model
from modules.ingestion_prepro import ingestion_preprocess
from modules.valid import validate
from modules.wrapper import NeuralODE

# Models f'
from models_derivative.mlp import mlp


# Import and preprocessing data
train_dataset, test_dataset = ingestion_preprocess(amplitudes=['050'], seq_size=3000, train_percentage=0.9) # 9900 unique seq

# Models 
model = NeuralODE(f_prime_model= mlp(), integrator='euler')

train_model(train_dataset, model, 100, path= 'skibiditoiler', batch_size=1, scheduler=False, learning_rate=0.01, print_every=1)

validate(model, test_dataset, num_traj=1)



