import numpy as np
import jax.numpy as jnp
import h5py
import math


from modules.utils import downsampling, random_split


# In principle there is observation every 1e-6s ----> 1e-4 every time step with downsampling 
# ----> take 100 time steps before i.e. let's say from 0s to 1e-2s (10 ms, physics time of restoration is 8ms)
# ----> of these 100 time steps, downsample them again of a factor of 3 (take one every 3)
# ----> Basically one time step every 3e-4

def ingestion_preprocess(run):

    amplitudes=run['amplitudes']
    timeHistorySizeOfU=100
    downsampling_inputs=run['input_downsample_factor']
    train_percentage=run['train_percentage']
    downsampling_strat=run['downsampling_strat']

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
    input_data = np.stack(input_data_list).squeeze(0)
    output_data = np.stack(output_data_list).squeeze(0)
    time_data = np.stack(time_data_list).squeeze(0)

    # Adjust inputs and times
    input_data = np.stack((time_data, input_data)).swapaxes(0,1)
    time_data = time_data[:, 0]
    last_element = input_data[:, 0, -1]
    last_element = last_element[:, np.newaxis]
    input_data[:, 0, :] = - (input_data[:, 0, :] - last_element)

    print(f"Time data: {time_data.shape}")
    print(f"Input data: {input_data.shape}")
    print(f"Output data: {output_data.shape}")


    # Split the dataset
    train_dataset, test_dataset = random_split(time_data, input_data, output_data, split_ratio=train_percentage)

    return train_dataset, test_dataset, input_size


