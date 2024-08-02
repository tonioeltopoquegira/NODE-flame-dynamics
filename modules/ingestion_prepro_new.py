import numpy as np
import h5py
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset, random_split
import math

from scipy.interpolate import interp1d


def ingestion_preprocess(amplitudes, timeHistorySizeOfU=100, downsampling_factor=3, seq_size = 500, train_percentage=0.6):

    print(f"""
        Explanation Dataset 
        -------------------------
        In principle there is observation every {1e-6*1000} ms 
        ---> observation every {1e-6*100*1000} ms with default downsampling
        ---> take window of {timeHistorySizeOfU} time steps before for each obs
            i.e. a window of {1e-4*timeHistorySizeOfU*1000} ms before the heat release value (physics time of restoration is 8ms)
        ---> of these {timeHistorySizeOfU} time steps, downsample them again of a factor of {downsampling_factor} (take one every {downsampling_factor})
        ---> Basically one time step every {downsampling_factor*1e-6*100*1000} ms for a total of {math.ceil(timeHistorySizeOfU/downsampling_factor)} time steps

        """)

    hf = h5py.File('data/Kornilov_Haeringer_all.h5', 'r')

    input_size = math.ceil(timeHistorySizeOfU/downsampling_factor)

    # Initialize lists to hold the split data
    input_data_list = []
    output_data_list = []
    time_data_list = []

    for a in amplitudes:
        output_data = np.array(hf.get('BB_A' + a+ '_Q'))
        input_data = np.array(hf.get('BB_A' + a+ '_U'))
        time_data = np.array(hf.get('BB_time'))

        # Downsample the data - To do for every datasets
        output_data = output_data[0::100] 
        input_data = input_data[0::100]
        time_data = time_data[0::100]

        # Cut off data for which we don't have enough inputs
        output_data = output_data[timeHistorySizeOfU-1:] # FOR SOME REASON #TODO

        # Reshape the data using sliding window view
        input_data = np.lib.stride_tricks.sliding_window_view(input_data, timeHistorySizeOfU)
        time_data = np.lib.stride_tricks.sliding_window_view(time_data, timeHistorySizeOfU)
        
        input_data = input_data[:,::downsampling_factor]
        time_data = time_data[:,::downsampling_factor]
        
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
    time_tensor = torch.tensor(time_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)

    input_tensor = torch.stack((time_tensor, input_tensor))
    time_tensor = time_tensor[:, :, 0]

    # Create Initial Values tensor  
    iv_tensor = output_tensor[:, 0]

    input_tensor = input_tensor.permute(1,0,2,3)

    # Create a TensorDataset
    dataset = TensorDataset(time_tensor, input_tensor, output_tensor, iv_tensor)

    # Calculate the sizes for train, validation, and test sets
    total_samples = len(dataset)
    train_size = math.ceil(total_samples * (train_percentage))
    test_size = total_samples - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset, input_size

"""class Interpolator(torch.nn.Module):
    
    def __init__(self, x, y, mode):
        super(Interpolator, self).__init__()
        self.x = x.unsqueeze(0).unsqueeze(0)
        self.y = y.unsqueeze(0).unsqueeze(0)

    def forward(self, t):
        return torch.interp(t, self.x, self.y)"""