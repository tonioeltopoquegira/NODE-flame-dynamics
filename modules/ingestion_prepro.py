import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, random_split
import math
from modules.utils import downsampling


def ingestion_preprocess(amplitudes, timeHistorySizeOfU=100, downsampling_inputs=3, seq_size = 500, train_percentage=0.6, downsampling_strat = 'uniform'):

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

        # Downsample the data - To do for every datasets #TODO Are we sure we used it for the NODE?
        time_data, input_data, output_data = downsampling(time_data, input_data, output_data, downsampling_factor=100, mode=downsampling_strat)

        # Cut off data for which we don't have enough inputs
        output_data = output_data[timeHistorySizeOfU-1:] # FOR SOME REASON #TODO

        # Reshape the data using sliding window view
        input_data = np.lib.stride_tricks.sliding_window_view(input_data, timeHistorySizeOfU)
        time_data = np.lib.stride_tricks.sliding_window_view(time_data, timeHistorySizeOfU)
        
        input_data = input_data[:,::downsampling_inputs]
        time_data = time_data[:,::downsampling_inputs]
        
        # Split big unique sequences into smaller sequences (trajectories) of size seq_size
        num_sequences = input_data.shape[0] // seq_size

        for i in range(num_sequences):

            start_idx = i * seq_size
            end_idx = start_idx + seq_size

            # Extract the smaller sequences
            input_seq = input_data[start_idx:end_idx]
            output_seq = output_data[start_idx:end_idx]
            time_seq = time_data[start_idx:end_idx]

            # Append the sequences to the respective lists
            input_data_list.append(input_seq)
            output_data_list.append(output_seq)
            time_data_list.append(time_seq)


    # Concatenate the data from all amplitudes
    input_data = np.stack(input_data_list)
    output_data = np.stack(output_data_list)
    time_data = np.stack(time_data_list)

    # Convert to torch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    time_tensor = torch.tensor(time_data, dtype=torch.float32)
    input_tensor = torch.stack((time_tensor, input_tensor))
    time_tensor = time_tensor[:, :, 0]

    # Adjust inputs and times
    last_element = input_tensor[:, 0, :, -1]
    last_element = last_element[:, :, np.newaxis]
    input_tensor[:, 0, :, :] = - (input_tensor[:, 0, :, :] - last_element)
    # Create Initial Values tensor  
    iv_tensor = output_tensor[:, 0]

    # Permute and show shapes
    input_tensor = input_tensor.permute(1,0,2,3)
    print(time_tensor.shape)
    print(input_tensor.shape)
    print(output_tensor.shape)
    print(iv_tensor.shape)
    
    # Create a TensorDataset
    dataset = TensorDataset(time_tensor, input_tensor, output_tensor, iv_tensor)

    # Calculate the sizes for train, validation, and test sets
    total_samples = len(dataset)
    train_size = math.ceil(total_samples * (train_percentage))
    test_size = total_samples - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset, input_size
