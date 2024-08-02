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
from modules.update_params import update_weights
from modules.utils import set_up_folders
from modules.wrapper import NeuralODE

# Models f'
from models_derivative.mlp import mlp


# Import and preprocessing data
train_dataset, test_dataset, input_size = ingestion_preprocess(amplitudes=['050'], seq_size=1000, train_percentage=0.9, downsampling_factor=1, timeHistorySizeOfU=34) # 9900 unique seq

# Choices
model_name = 'mlp_less_history'
model = mlp(input_size)
integrator = 'euler'

path = set_up_folders(model_name, integrator)

# Models 
model = NeuralODE(f_prime_model= model, integrator=integrator)

if os.path.exists(f"weights/{path}/weights.pth"):
    print("Loaded existing weights...")
    model = update_weights(model, path)
    # We could learn using easy integrator then plug our derivative in better one and use a interpolator (no need to backpropagate through it)


train_model(train_dataset, model, 40, path=path, batch_size=1, scheduler=False, learning_rate=0.005, print_every=1)

# Update Model
model = update_weights(model, path)




