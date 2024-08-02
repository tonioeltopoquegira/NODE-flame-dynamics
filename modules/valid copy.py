import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from models.kan import B_splines, Cheby

# Validation file

def validate(model, X_train, Y_train, X_val, Y_val, name, path):

    model.eval()

    train_loss = pred_plot(model, X_train, Y_train, "training", name, path)
    test_loss = pred_plot(model, X_val, Y_val, "testing", name, path)

    if "kan" in name and "gru" in name:
        model.kan_in.draw_bases(path)
        model.kan_in.draw_activations(path)
        model.kan_hid.draw_activations(path)
    
    elif "kan" in name:
        model.draw_bases(path = path)
        model.draw_activations(path=path)
    
    return train_loss.item(), test_loss.item()

def compute_rmse(y_pred, y_true):
    mse = torch.mean((y_pred - y_true)**2)
    mean_squared_true = torch.mean(y_true**2)
    rmse = torch.sqrt(mse / mean_squared_true)
    return rmse

def pred_plot(model, x, y, title, name, path):
    x = x[:1000:2,:]
    y = y[:1000:2,:]
    with torch.no_grad():
        pred = model(x)
        
    loss = compute_rmse(pred, y)

    plt.plot(pred, linestyle='dashed', label='Predicted')
    plt.plot(y, label='Actual')
    plt.ylabel('Heat Rate')
    plt.grid()
    plt.xlabel('Time')
    plt.title(f"Prediction on {title} data")
    plt.legend()
    """plt.text(0.5, 0.9, f'Loss: {loss.item():.4f}', transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))"""
    plt.savefig(f'figures/{path}/{title}.png')
    plt.close()

    return loss

def draw_bases(kan, name, path):
    x = torch.linspace(-1,1, 100).unsqueeze(-1).repeat(1, 1, kan.kan_lays[0].in_size).squeeze(0) # 2 chosen randomly to be fair... ensure correct broadcasting
    x = x.transpose(0,1)
    if kan.fun=="spline":
        bases = B_splines(x, kan.grid, kan.k).permute(2,0,1).unsqueeze(2) # WHY WE CAN'T LEAVE IT INSIDE the FUNCTION :(
    elif kan.fun == "cheby":
        bases = Cheby(x, kan.k)

    for b in range(bases.shape[-1]):
        plt.plot(torch.linspace(-1,1,100), bases[:,0,0, b], label=f'Base {b}')
    plt.grid()
    plt.title(f"Bases for chosen grid and order {kan.k}")
    plt.legend()
    plt.savefig(f"figures/{path}/basis_splines.png")
    plt.close()

def draw_activations(kan, lay, input_pos, out_pos, name, path):

    # generate b-splines for 100 points
    x = torch.linspace(-1,1, 100).unsqueeze(-1).repeat(1, 1, kan.kan_lays[0].in_size).squeeze(0) # 2 chosen randomly to be fair... ensure correct broadcasting
    x = x.transpose(0,1)
    if kan.fun == "spline":
        bases = B_splines(x, kan.grid, kan.k)
    if kan.fun == "cheby":
        bases = Cheby(x, kan.k).squeeze(2).permute(1,2,0)

    # extract coefficients
    coeff = torch.cat([param.unsqueeze(0) for param in kan.kan_lays[lay].fun_weights], dim=0).unsqueeze(-1)
    coeff = coeff[:, input_pos, out_pos, :]
    partial_output = torch.sum(bases * coeff, 1).detach().numpy()

    plt.plot(torch.linspace(-1,1,100), partial_output[0,:])
    plt.grid()
    plt.title(f"Activation function for the layer {lay} ({input_pos, out_pos})")
    plt.savefig(f"figures/{path}/activations/layer{lay}({input_pos}_{out_pos}).png")
    plt.close()
    


    
    





    
    


