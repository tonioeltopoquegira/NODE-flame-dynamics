import numpy as np
import h5py
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset, random_split
import math
import matplotlib.pyplot as plt

from modules.utils import downsampling

# In principle there is observation every 1e-6s ----> 1e-4 every time step with downsampling 
# ----> take 100 time steps before i.e. let's say from 0s to 1e-2s (10 ms, physics time of restoration is 8ms)
# ----> of these 100 time steps, downsample them again of a factor of 3 (take one every 3)
# ----> Basically one time step every 3e-4

def ingestion_preprocess(amplitudes, timeHistorySizeOfU=100, downsampling_inputs=3, train_percentage=0.9, downsampling_strat = 'uniform'):

    hf = h5py.File('data/Kornilov_Haeringer_all.h5', 'r')

    input_size = math.ceil(timeHistorySizeOfU/downsampling_inputs)

    # Initialize lists to hold the split data
    input_data_list = []
    output_data_list = []
    time_data_list = []

    
    for a in amplitudes:
        output_data = np.array(hf.get('BB_A' + a+ '_Q'))
        input_data = np.array(hf.get('BB_A' + a+ '_U'))
        time_data = np.array(hf.get('BB_time'))

        # Downsample the data - To do for every datasets
        time_data, input_data, output_data = downsampling(time_data, input_data, output_data, downsampling_factor=100, mode=downsampling_strat)

        # Cut off data for which we don't have enough inputs
        output_data = output_data[timeHistorySizeOfU-1:] # FOR SOME REASON #TODO

        # Reshape the data using sliding window view
        input_data = np.lib.stride_tricks.sliding_window_view(input_data, timeHistorySizeOfU)
        time_data = np.lib.stride_tricks.sliding_window_view(time_data, timeHistorySizeOfU)
        
        input_data = input_data[:,::downsampling_inputs]
        time_data = time_data[:,::downsampling_inputs]
        
        # Append the sequences to the respective lists
        input_data_list.append(input_data)
        output_data_list.append(output_data)
        time_data_list.append(time_data)

    # Concatenate the data from all amplitudes
    input_data = np.stack(input_data_list)
    output_data = np.stack(output_data_list)
    time_data = np.stack(time_data_list)

    # Convert to torch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    time_tensor = torch.tensor(time_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)

    # Adjust inputs and times
    input_tensor = torch.stack((time_tensor, input_tensor))
    last_element = input_tensor[:, 0, :, -1]
    last_element = last_element[:, :, np.newaxis]
    input_tensor[:, 0, :, :] = - (input_tensor[:, 0, :, :] - last_element)

    # Permute and show shapes
    input_tensor = input_tensor.squeeze(1).permute(1, 0,2)
    output_tensor = output_tensor.transpose(0,1)
    
    # Create a TensorDataset
    dataset = TensorDataset(input_tensor, output_tensor)

    # Calculate the sizes for train, validation, and test sets
    total_samples = len(dataset)
    train_size = math.ceil(total_samples * (train_percentage))
    test_size = total_samples - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset, input_size


