import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# Validation file

def validate(model, dataset, num_traj):
    for (x, y, iv), en in zip(dataset, range(num_traj)):
        pred = model(x, iv)
        plt.plot(pred)
        plt.plot(y)
        plt.show()