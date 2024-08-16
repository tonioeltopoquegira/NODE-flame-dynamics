import numpy as np
import h5py
import torch
import math
from modules.utils import downsampling, random_split
from models.node_utils.interpolator import Interpolator1D


def ingestion_preprocess(run):

    """ Preprocess the data before training for the Neural ODE model. 

    Returns:
       train_dataset (list of jnp.arrays) -> training dataset including time points, time points at which we interpolate 
                                            for u to approximate the derivative, output, initial values 
       test_data (list of jnp.array) -> testing dataset including time points, time points at which we interpolate 
                                            for u to approximate the derivative, output, initial values 
       input_size (int) -> number of evaluations of u(t) to feed in the neural net to approximate derivative
       interpol (nn.module) -> Interpolator to evaluate u(t) from any instance of t
    """

    amplitudes= run['amplitudes']
    seq_size=run['seq_size']
    train_percentage=run['train_percentage']
    downsampling_inputs=run['input_downsample_factor']
    timeHistorySizeOfU=100
    downsampling_strat= run['downsampling_strat']

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

        interpol = Interpolator1D(time_data, input_data, 'linear')

        # Downsample the data - To do for every datasets #TODO Are we sure we used it for the NODE?
        time_data, input_data, output_data = downsampling(time_data, input_data, output_data, downsampling_factor=100, mode=downsampling_strat)

        # Cut off data for which we don't have enough inputs
        output_data = output_data[timeHistorySizeOfU-1:] # FOR SOME REASON #TODO

        # Reshape the data using sliding window view
        input_data = np.lib.stride_tricks.sliding_window_view(time_data, timeHistorySizeOfU)
        
        input_data = input_data[:,::downsampling_inputs]
        #time_data = time_data[:,::downsampling_inputs]
        
        # Split big unique sequences into smaller sequences (trajectories) of size seq_size
        num_sequences = len(input_data) // seq_size

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
    # input_data = np.stack((time_data, input_data)).swapaxes(0, 1)

    # Adjust inputs and times
    last_element = input_data[:, :, -1]
    last_element = last_element[:, :, np.newaxis]
    input_data[:, :, :] = - (input_data[:, :, :] - last_element)
    #input_data = input_data[:, 0, :, :] # only time steps to evaluate function!

    # Create Initial Values tensor  
    iv_data = output_data[:, 0]

    # Permute and show shapes
    print(f"Times: {time_data.shape}")
    print(f"Input Data: {input_data.shape}")
    print(f"Output Data: {output_data.shape}")
    print(f"Initial Values: {iv_data.shape}")
    
    # Split the dataset - Error here
    train_dataset, test_dataset = random_split(time_data, input_data, output_data, iv_data, split_ratio= train_percentage)

    return train_dataset, test_dataset, input_size, interpol
