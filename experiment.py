# General Imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Jax imports
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import orbax_utils

# Functions
from modules.train import train_model
from modules.ingestion_prepro import ingestion_preprocess as ingestion_preprocess_node
from modules.ingestion_prepro_other import ingestion_preprocess as ingestion_preprocess_other
from modules.valid import validate
from modules.utils import set_up_folders, record_run_from_file
from modules.bookeeping_params import init_params, restore_params

# Models 
#from models.deriv_models.mlp import mlp as fprime_mlp
from models.node import NeuralODE
from models.mlp import mlp


def run_experiment(run):
    
    path = set_up_folders(run)

    if run['model'] in ['node', 'gru']:
        train_dataset, test_dataset, input_size = ingestion_preprocess_node(run)
    else:
        train_dataset, test_dataset, input_size = ingestion_preprocess_other(run)

    if run['model'] == 'node':
        # create integrator, interpolator and model for fprime
        model = NeuralODE(f_prime_model=fprime_mlp(input_size), integrator='euler')

    elif run['model'] == 'mlp':
        model = mlp(input_sizes=[input_size] + run['hidd_sizes'], nonlinearity=run['nonlinearity'], time_dependency=run['time-dep'], time_sizes=run['time_hidd_sizes'], initializer=run['initializer'])

    if os.path.exists(f"weights/{path}") and not run['from_scratch']:

        # Update the weights
        print("Loading existing weights...")

        params, mngr = restore_params(path)
    
    else:
        print("Training from scratch...")
        params, mngr = init_params(model, train_dataset, path)
        
    # Train from starting of params
    train_model(train_dataset, params, model, run, path, mngr)

    # Validation
    validate(run, model, path, test_dataset, mngr)

    record_run_from_file(run, 'run_report.xlsx', 'run_report.txt')

if __name__ == '__main__':
    
    run = {
    # Dictionaries and other stuff
    'save_dict': False,
    'dict': 'runs_report.xlsx',
    'from_scratch' : False,
    'show_res': False,

    # Data preprocessing
    'amplitudes': ['050'],
    'seq_size': 500,
    'train_percentage': 0.9,
    'input_downsample_factor': 3,
    'downsampling_strat': 'uniform',
    'interpolator': 'linear', #     (*)

    # Model
    'model_name' : 'mlp_lin',
    'model': 'mlp',
    'integ_strat': 'fixed-grid', # ['fixed-grid', 'adaptive']   (*)
    'integ_method': 'rk4', # ['euler', 'rk4']   (*)
    'deriv_model': 'mlp', # [ 'mlp' ]   (*)
    'hidd_sizes': [1],
    'nonlinearity': 'tanh', # ['tanh', 'relu', 'sigmoid', 'softplus', 'elu', 'selu', 'swish']
    'time-dep':'none', # ['time-att', 'time-branch']
    'time_hidd_sizes': [1,3,1],
    'initializer': 'xavier',

    # Training
    'epochs': 200,
    'batch_size': 3000, # (**)
    'learning_rate': 0.001,
}

# (*) APPLICABLE ONLY IF MODEL IS 'node'
# (**)  atm, must be 1 if model is 'node'

    run_experiment(run)