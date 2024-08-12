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
from modules.ingestion_prepro import ingestion_preprocess as ingestion_preprocess_node
from modules.ingestion_prepro_other import ingestion_preprocess as ingestion_preprocess_other
from modules.valid import validate
from modules.update_params import update_weights
from modules.utils import set_up_folders
from modules.valid import validate


# Models 
from models.deriv_models.mlp import mlp as fprime_mlp
from models.NODE_wrapper import NeuralODE
from models.mlp import mlp

model_name = 'mlp_comparison'
path = set_up_folders(model_name)
retrain = True

# Import and preprocessing data
if 'node' in model_name:
    train_dataset, test_dataset, input_size = ingestion_preprocess_node(amplitudes=['050'], seq_size=1000, train_percentage=0.9, downsampling_inputs=3, timeHistorySizeOfU=100, downsampling_strat= 'uniform') # 9900 unique seq
else:
    train_dataset, test_dataset, input_size = ingestion_preprocess_other(amplitudes=['050'], timeHistorySizeOfU=100, downsampling_inputs=3, train_percentage=0.9, downsampling_strat='uniform')

#model = NeuralODE(f_prime_model=fprime_mlp(input_size), integrator='euler')
model = mlp(input_sizes=[input_size, 10, 1], nonlinearity='tanh', time_dependency='none', time_sizes=[1, 3, 1])
if os.path.exists(f"weights/{path}/weights.pth") and not retrain:
    # Update the weights
    print("Loading existing weights...")
    model = update_weights(model, path)

    # We could learn using easy integrator then plug our derivative in better one and use a interpolator (no need to backpropagate through it)

print(model)
train_model(train_dataset, model, 200, model_name = model_name, path=path, batch_size=600, scheduler=False, learning_rate=0.001, print_every=1)

# Update Model
model = update_weights(model, path)

validate(model, model_name, path, test_dataset, num_traj=2)