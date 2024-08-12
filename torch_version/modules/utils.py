import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import jax.numpy as jnp

folder_names = {
    ('050',) : 'Low_Amp',
    ('150',) : 'High_Amp',
    ('100',) : 'Med_Amp',
    ('050', '100', '150'): 'All_Amp',
    ('050', '100'): '100_50_Amp'
}

def write_summary(excel_filename, model_name, data_details, train_details, train_loss, test_loss, metrics_harmonics):
    run = {'model': model_name}
    run.update(data_details)
    run.update(train_details)
    run.update({
    'rMSE_train_loss': f'{train_loss:.3}', 
    'rMSE_test_loss': f'{test_loss:.3}'
    })
    run.update(metrics_harmonics)


    df = pd.DataFrame([run])

    if os.path.isfile(excel_filename):
        df_existing = pd.read_excel(excel_filename)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df

    summ = os.path.join("summaries", excel_filename)
    df_combined.to_excel(summ, index=False)

    print(f"Summary has been written to {excel_filename}!")

def set_up_folders(model_name):
    os.makedirs('figures/downsampling', exist_ok=True)
    path = os.path.join(model_name)
    directory_path = os.path.join("figures", path)
    os.makedirs(directory_path, exist_ok=True)
    if 'node' in model_name:
        os.makedirs(directory_path+'/training_process', exist_ok=True)
    os.makedirs(directory_path+'/test_results', exist_ok=True)
    directory_path = os.path.join("weights",path)
    os.makedirs(directory_path, exist_ok=True)
    return path


# DATA DOWNSAMPLING

def graph_sampling(time_data, output_data, time_data_downsampled, output_data_downsampled, mode):

    time_data_downsampled = time_data_downsampled[:1000]
    output_data_downsampled = output_data_downsampled[:1000]

    indeces = np.where((time_data <= time_data_downsampled[-1]) & (time_data >= time_data_downsampled[0]))
    time_data = time_data[indeces]
    output_data = output_data[indeces]


    # Plot the original output data over time
    plt.figure(figsize=(18, 6))
    plt.plot(time_data, output_data, color='gray')

    # Highlight the downsampled points with red dots and connect them with a different color line
    plt.plot(time_data_downsampled, output_data_downsampled, 'o', color='red', markersize=1)

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Output Data')
    plt.title('Output Data Over Time Downsampling')

    # Show the plot
    plt.savefig(f'figures/downsampling/{mode}.png')
    plt.close()

# Another possibility is to focus on where derivatives are the highest!

def downsampling(time_data, input_data, output_data, downsampling_factor = 100, mode = 'uniform'):


    if mode == "non-uniform-1":
        # Calculate importance scores
        importance_scores = np.abs(output_data)

        # Normalize the importance scores to form a probability distribution
        probability_distribution = importance_scores / np.sum(importance_scores)

        # Create the cumulative distribution function (CDF)
        cdf = np.cumsum(probability_distribution)

        # Determine the number of samples to keep
        num_samples = math.ceil(len(output_data) / downsampling_factor)

        # Use inverse transform sampling to select indices based on the CDF
        random_values = np.random.rand(num_samples)
        selected_indices = np.searchsorted(cdf, random_values)

        # Downsample the input, output, and time data
        input_data_downsampled = input_data[selected_indices]
        output_data_downsampled = output_data[selected_indices]
        time_data_downsampled = time_data[selected_indices]

        sorted_indices = np.argsort(time_data_downsampled)
        time_data_new = time_data_downsampled[sorted_indices]
        output_data_new = output_data_downsampled[sorted_indices]
        input_data_new = input_data_downsampled[sorted_indices]
    
    if mode == "non-uniform-2":
        # Calculate importance scores
        importance_scores = np.sqrt(np.abs(output_data))

        # Normalize the importance scores to form a probability distribution
        probability_distribution = importance_scores / np.sum(importance_scores)

        # Create the cumulative distribution function (CDF)
        cdf = np.cumsum(probability_distribution)

        # Determine the number of samples to keep
        num_samples = math.ceil(len(output_data) / downsampling_factor)

        # Use inverse transform sampling to select indices based on the CDF
        random_values = np.random.rand(num_samples)
        selected_indices = np.searchsorted(cdf, random_values)

        # Downsample the input, output, and time data
        input_data_downsampled = input_data[selected_indices]
        output_data_downsampled = output_data[selected_indices]
        time_data_downsampled = time_data[selected_indices]

        sorted_indices = np.argsort(time_data_downsampled)
        time_data_new = time_data_downsampled[sorted_indices]
        output_data_new = output_data_downsampled[sorted_indices]
        input_data_new = input_data_downsampled[sorted_indices]

    
    if mode == "uniform":
        output_data_new = output_data[::downsampling_factor]
        input_data_new = input_data[::downsampling_factor]
        time_data_new = time_data[::downsampling_factor]

    
    graph_sampling(time_data, output_data, time_data_new, output_data_new, mode)

    return time_data_new, input_data_new, output_data_new


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl



def random_split(*arrays, split_ratio=0.8, seed=None):
    """
    Randomly splits a list of arrays into training and validation sets along the first dimension.

    Args:
        *arrays (np.ndarray): A list of arrays with the same shape along the first dimension.
        split_ratio (float): The proportion of the data to be used for training.
        seed (int, optional): Seed for the random number generator.

    Returns:
        Tuple of lists of split arrays: (train_arrays, val_arrays).
    """
    num_samples = arrays[0].shape[0]
    
    # Ensure all arrays have the same shape along the first dimension
    for array in arrays:
        if array.shape[0] != num_samples:
            raise ValueError("All arrays must have the same shape along the first dimension.")

    indices = np.arange(num_samples)

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split indices
    split_index = int(num_samples * split_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    # Split each array according to the indices
    train_arrays = [jnp.array(array[train_indices]) for array in arrays]
    val_arrays = [jnp.array(array[val_indices]) for array in arrays]

    return train_arrays, val_arrays
