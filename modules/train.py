# Imports 
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dataset, model, epochs, path, batch_size = 3, scheduler = False, learning_rate = 1e-5, print_every=1):
    
    # PyTorch Dataset and DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # batch size > 200

    # optimizer & scheduler
    
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    if scheduler:
        """
        Adam can substantially benefit from a scheduled learning rate multiplier. The fact that Adam
        is an adaptive gradient algorithm and as such adapts the learning rate for each parameter
        does not rule out the possibility to substantially improve its performance by using a global
        learning rate multiplier, scheduled, e.g., by cosine annealing.

        https://discuss.pytorch.org/t/with-adam-optimizer-is-it-necessary-to-use-a-learning-scheduler/66477/2
        """
        # This is not very useful with Adam! May be conflicting
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=600, T_mult=10)
        pass

    # criterion
    criterion = nn.MSELoss()
    
    # Summaries, Parameters, Model Complexity
    #run_entry = summary_training(name, model, input_size, epochs, scheduler, att)

    """print(f"Initial Weights:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
"""
    print("Training...")

    loss_train = np.zeros(epochs)

    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        #for i, (inputs, targets, iv) in enumerate(data_loader):
        for i, (times, inputs, targets, iv) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            
            # Forward pass
            
            pred = model(times, inputs, iv)
            if i == 0:
                train_plot_pred(inputs, targets, pred, path, epoch)

            loss = criterion(pred, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if scheduler:
            scheduler.step()
            #print(scheduler.get_lr())

        # Print epoch statistics
        if (epoch+1) % print_every == 0:
            print(f"Epoch {epoch+1}, Epoch Total MSE: {(epoch_loss):.3f}, Epoch Time: {epoch_time:.2f}s")
        
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), f'weights/{path}/weights.pth')
        
        loss_train[epoch] = epoch_loss

        """print(f"Epoch {epoch + 1} Weights:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data}")"""
    # Save model weights
    torch.save(model.state_dict(), f'weights/{path}/weights.pth')
    plt.plot(np.arange(epochs), loss_train)
    plt.title("Training Loss")
    plt.savefig(f"figures/{path}/training_loss.png")
    plt.close()


def summary_training(name, model, input_size, epochs, scheduler, att):

    # Summaries, Parameters, Model Complexity

    if "gru" in name and "kan" in name:
        run_entry = {
            'n_params' : 14,
            'architec' : "gru_kan!"
        }
    
    elif "ckan" in name:
        tot_param = 0
        coeff = model.k 
        print(f"KAN Parameter Count \n Poly order {model.k}; Grid_Size {model.grid_size} \n ======================")
        for i, (layin, layout) in enumerate(zip(model.lay_sizes[:-1], model.lay_sizes[1:])):
            print(f"Coefficients {i} layer: {layin} x {coeff} x {layout}")
            tot_param += layin * coeff * layout
        print("+ 2 Shared Linear Parameters\n ======================")
        print(f"Tot: {tot_param+2}")

        run_entry = {
            'n_params' : tot_param,
            'architec' : model.lay_sizes
        }

    elif "kan" in name:
        tot_param = 0
        coeff = model.k + model.grid_size - 1
        print(f"KAN Parameter Count \n Poly order {model.k}; Grid_Size {model.grid_size} \n ======================")
        for i, (layin, layout) in enumerate(zip(model.lay_sizes[:-1], model.lay_sizes[1:])):
            print(f"Coefficients {i} layer: {layin} x {coeff} x {layout}")
            tot_param += layin * coeff * layout
        print("+ 2 Shared Linear Parameters\n ======================")
        print(f"Tot: {tot_param+2}")

        run_entry = {
            'n_params' : tot_param,
            'architec' : model.lay_sizes
        }

    else:
        sum = summary(model, input_size=(1,input_size))

        run_entry = {
            'n_params' : sum.total_params,
            'architec' : model,
        }
    
    run_entry.update(
        {'epochs' : epochs,
        'scheduler': scheduler,
        'att': att})

    return run_entry

def train_plot_pred(inputs, targets, pred, path, epoch):
    targets = targets.squeeze(0).squeeze(0).squeeze(-1)
    pred = pred.squeeze(0).squeeze(0).squeeze(-1).squeeze(-1)

    times = inputs[:,1,:,0].squeeze(0).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    plt.plot(times, targets, label='Ground Truth', linestyle='--', color='k')
    plt.plot(times, pred, label='Prediction', linestyle='-', color='r')
    plt.title(f"Prediction at epoch {epoch+1}")
    plt.legend()
    plt.savefig(f"figures/{path}/training_epoch_{epoch+1}.png")
    plt.close()