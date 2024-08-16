import os
import orbax.checkpoint as ocp
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import shutil


def init_params(model, run, dataset, path):

    # Create Checkpoint Manager
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    save_path = os.path.abspath(f"weights/{path}") 

    # Create a Manager
    mngr = ocp.CheckpointManager(save_path, checkpointer, options)

    rng = random.PRNGKey(run['seed'])
    shape_input = dataset[1].shape[1:]

    # Initialize the RNG and model parameters
    if run['model'] == 'node':
        shape_times = dataset[0].shape[1:]
        shape_iv = dataset[3].shape[1:]
        params = model.init(rng, jnp.arange(0, shape_times[0], 1, dtype=jnp.float32), jnp.ones(shape_input), jnp.ones(shape_iv))['params']
    else:
        
        dummy_batch_size = (10000,)
        shape_input = dummy_batch_size + shape_input
        params = model.init(rng, jnp.ones(shape_input))['params']
        
    
    
    # Delete the previous stuff
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    return params, mngr

def restore_params(path):

    # Create Checkpoint Manager
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    save_path = os.path.abspath(f"weights/{path}") 

    # Create a Manager
    mngr = ocp.CheckpointManager(save_path, checkpointer, options)
    restored_ckpt = mngr.restore(mngr.latest_step())
    params = restored_ckpt["state"]["params"]
    
    return params, mngr