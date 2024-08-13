import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import jax.numpy as jnp
import jax
from flax.training import train_state
import orbax.checkpoint as ocp
from jax import random
import shutil


# CODE FLOW UTILITIES

def record_run_from_file(run, excel_file='run_report.xlsx', text_file='run_report.txt'):
    """
    Reads a run configuration from a file, generates a new key, and records it in both an Excel file and a text file.

    Args:
        run_file (str): Path to the file containing the run configuration (JSON or similar format).
        excel_file (str): Name of the Excel file to record the run configurations.
        text_file (str): Name of the text file to save the run configuration as text.
    """

    excel_file = os.path.join('summaries', excel_file)
    text_file = os.path.join('summaries', text_file)

    # Generate a new key
    key = get_next_key(excel_file)
    
    # Create a DataFrame from the run dictionary
    df = pd.DataFrame([run])
    df['run_key'] = key

    # Check if the Excel file already exists
    if os.path.exists(excel_file):
        # Load existing data and append new data
        existing_df = pd.read_excel(excel_file, engine='openpyxl')
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Create a new DataFrame if the file does not exist
        combined_df = df

    # Save the updated DataFrame to the Excel file
    combined_df.to_excel(excel_file, index=False, engine='openpyxl')

    # Save the run dictionary as a text entry
    with open(text_file, 'a') as file:
        file.write(f"Run Key: {key}\n")
        file.write(f"Configuration: {run}\n\n")
    
    print(f"Summaries written to {excel_file} and {text_file}!")

    
def get_next_key(excel_file='runs_report.xlsx'):
    """
    Gets the next available key for the run by checking the number of existing entries.
    
    Args:
        excel_file (str): Name of the Excel file to check for existing entries.
        
    Returns:
        str: The next available key.
    """
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file, engine='openpyxl')
        num_entries = len(df)
    else:
        num_entries = 0
    
    return f"run_{num_entries + 1:03d}"

def set_up_folders(run):
    
    model_name = run['model_name']

    path = os.path.join(model_name)
    directory_path = os.path.join("figures", path)
    os.makedirs(directory_path, exist_ok=True)
    if run['model'] == 'node':
        os.makedirs(directory_path+'/training_evolution', exist_ok=True)
    if run['time-dep'] == 'att':
       os.makedirs(directory_path+'/attention_evolution', exist_ok=True)
    os.makedirs(directory_path+'/test_results', exist_ok=True)
    directory_path = os.path.join("weights",path)
    os.makedirs(directory_path, exist_ok=True)
    return path


# DATA PROCESSING

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


# TRAINING

def data_loader(*arrays, batch_size):
    """
    A simple data loader for batching data arrays.

    Args:
        arrays: One or more NumPy arrays with the same first dimension.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data before splitting into batches.

    Yields:
        Batches of data arrays.
    """
    # Ensure all arrays have the same number of samples
    n_samples = arrays[0].shape[0]
    for array in arrays:
        assert array.shape[0] == n_samples, "All input arrays must have the same first dimension."
    
  
    indices = np.arange(n_samples)
    
    # Split into batches and yield
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)

def train_plot_pred(times, targets, pred, path, epoch):
    sorted_indices = jnp.argsort(times)
    times = times[sorted_indices]
    targets = targets[sorted_indices]
    pred = pred[sorted_indices]

    os.makedirs(f"figures/{path}/training_process", exist_ok=True)
    plt.figure(figsize=(16, 8))
    plt.plot(times[200:], targets[200:], label='Ground Truth', linestyle='--', color='k')
    plt.plot(times[200:], pred[200:], label='Prediction', linestyle='-', color='r')
    plt.title(f"Prediction at epoch {epoch}")
    plt.legend()
    plt.savefig(f"figures/{path}/training_process/{epoch}.png")
    plt.close()

def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

def mae_loss(pred, target):
    return jnp.mean(jnp.abs(pred-target))

# MODELS 

def choose_nonlinearity(name):
    """
    Returns the appropriate non-linearity function based on the given name.

    Args:
        name (str): The name of the non-linearity function.

    Returns:
        function: The corresponding non-linearity function.
    """
    if name == 'tanh':
        nl = jax.nn.tanh
    elif name == 'relu':
        nl = jax.nn.relu
    elif name == 'sigmoid':
        nl = jax.nn.sigmoid
    elif name == 'softplus':
        nl = jax.nn.softplus
    elif name == 'selu':
        nl = lambda x: jax.nn.selu(x)
    elif name == 'elu':
        nl = jax.nn.elu
    elif name == 'swish':
        nl = lambda x: x * jax.nn.sigmoid(x)
    else:
        raise ValueError("Nonlinearity not recognized")
    return nl

# Params Bookeeping

def init_updt_state(dataset, run, model, path, tx):

    # Create Checkpoint Manager
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    save_path = os.path.abspath(f"weights/{path}") 


    if os.path.exists(f"weights/{path}") and not run['retrain']:

        # Update the weights
        print("Loading existing weights...")

        # Create a Manager
        mngr = ocp.CheckpointManager(save_path, checkpointer, options)
        restored_ckpt = mngr.restore(mngr.latest_step())
        params = restored_ckpt["state"]["params"]
        
        # Restore a train state object
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
    
    else:
        # Initialize the RNG and model parameters
        shape_input = dataset[1].shape[1:]
        dummy_batch_size = (10000,)
        shape_input = dummy_batch_size + shape_input
        rng = random.PRNGKey(0)
        params = model.init(rng, jnp.ones(shape_input))['params']

        # Delete the previous stuff
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # Create Manager
        mngr = ocp.CheckpointManager(save_path, checkpointer, options)

        # Create a train state object
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    return state, mngr, params
