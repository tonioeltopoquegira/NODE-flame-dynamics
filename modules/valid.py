import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


# Validation file

def validate(model, model_name, path, dataset, num_traj):
    err = 0.0
    num_obs = 0

    if 'node' in model_name: 
        for (times, x, y, iv), en in zip(dataset, range(num_traj)):
            pred = model(times, x, iv)
            pred = pred.detach().numpy()
            y = y.detach().numpy()
            plt.figure(figsize=(16, 8))
            plt.plot(y, label='Ground Truth', linestyle='-', color='k')
            plt.plot(pred, label='Prediction', linestyle='-', color='r')
            plt.title("Prediction")
            plt.legend()
            plt.savefig(f"figures/{path}/test_results/traj_{en+1}")
            plt.close()
            err += np.sum((pred - y)**2)
            num_obs += len(y)
    
    else:
        data_load = DataLoader(dataset, 200, shuffle=False)
        for (x,y), en in zip(data_load, range(num_traj)):
            pred = model(x)
            pred = pred.detach().numpy()
            y = y.detach().numpy()
            plt.figure(figsize=(16, 8))
            plt.plot(y, label='Ground Truth', linestyle='-', color='k')
            plt.plot(pred, label='Prediction', linestyle='-', color='r')
            plt.title("Prediction")
            plt.legend()
            plt.savefig(f"figures/{path}/test_results/traj_{en+1}")
            plt.close()
            err += np.sum((pred - y)**2)
            num_obs += len(y)
        
        if 'att' in model_name:
            model.plot_attention(path, 999999)
        
    print(f"Mean Square Error Test Data: {err/num_obs}")
        

