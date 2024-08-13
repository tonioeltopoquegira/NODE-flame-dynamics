# Imports 
import time
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state, checkpoints, orbax_utils
from flax.core.frozen_dict import FrozenDict
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import orbax.checkpoint as ocp

from modules.utils import data_loader, train_plot_pred, mse_loss, mae_loss

DEVICE = jax.devices("gpu" if jax.lib.xla_bridge.get_backend().platform == "gpu" else "cpu")[0]


def train_model(dataset, params, model, run, path, mngr): 
    
    epochs = run['epochs']
    batch_size=run['batch_size']
    if run['model'] == 'node':
        batch_size = 1
    learning_rate=run['learning_rate']
    print_every=1
    
    # Optimizer
    if run['opt'] == 'adam':
        tx = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    elif run['opt'] == 'sgd':
        tx = optax.sgd(learning_rate=learning_rate)


    # Create a train state object
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    loss_train = np.zeros(epochs)

    print("Training...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for batch in tqdm(data_loader(*dataset, batch_size=batch_size), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            
            # Forward pass and loss computation
            
            if run['model'] == 'node':
                times, inputs, targets, ivs = batch
                def loss_fn(params):
                    t_eval, pred = model.apply({'params': params}, times, inputs, ivs)
                    loss = mse_loss(pred, targets)
                    return loss, pred
                
            else:
                times, inputs, targets = batch
                def loss_fn(params):
                    pred = model.apply({'params': params}, inputs)
                    loss = mse_loss(pred, targets)
                    return loss, pred


            # Backpropagation
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, preds) , grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            epoch_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print epoch statistics
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}, Epoch Total MSE: {(epoch_loss):.3f}, Epoch Time: {epoch_time:.2f}s")
        
        if (epoch+1)% 50 == 0:
            train_plot_pred(times, targets, preds, path, epoch)
        
        loss_train[epoch] = epoch_loss

        

    ckpt = {'state' : state}
    mngr.save(epoch, ckpt)
    plt.plot(np.arange(epochs), loss_train)
    plt.title("Training Loss")
    plt.savefig(f"figures/{path}/training_loss.png")
    plt.close()

    return mngr



