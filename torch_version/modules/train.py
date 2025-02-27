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

def train_model(dataset, model, epochs, model_name, path, batch_size = 3, scheduler = False, learning_rate = 1e-5, print_every=1):
    
    # PyTorch Dataset and DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # batch size > 200

    # optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    # criterion
    criterion = nn.MSELoss()
    
    # Summaries, Parameters, Model Complexity
    #run_entry = summary_training(name, model, input_size, epochs, scheduler, att)
    
    print("Training...")

    loss_train = np.zeros(epochs)

    torch.autograd.set_detect_anomaly(True)

    
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()


        for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")):

            if 'node' in model_name:
                times, inputs, targets, iv = batch
            else:
                inputs, targets = batch
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            if 'node' in model_name:
                pred = model(times, inputs, iv)
            else:
                pred = model(inputs)
            

            loss = criterion(pred, targets)


            # Backpropagation
            loss.backward()
        
            optimizer.step()
            
            epoch_loss += loss.item()
    
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        

        if scheduler:
            scheduler.step()

        # Print epoch statistics
        if (epoch+1) % print_every == 0:
            print(f"Epoch {epoch+1}, Epoch Total MSE: {(epoch_loss):.3f}, Epoch Time: {epoch_time:.2f}s")
        
        if (epoch+1) % 20 == 0 and 'node' in model_name:
            torch.save(model.state_dict(), f'weights/{path}/weights.pth')
        
        loss_train[epoch] = epoch_loss

    
    # Save model weights
    torch.save(model.state_dict(), f'weights/{path}/weights.pth')
    print(f"Params written to weights/{path}/weights.pth !")
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

def train_plot_pred(times, targets, pred, path, epoch):
    targets = targets.squeeze(0).squeeze(0).squeeze(-1)
    pred = pred.squeeze(0).squeeze(0).squeeze(-1).squeeze(-1)

    times = times.squeeze(0).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    sorted_indices = np.argsort(times)
    times = times[sorted_indices]
    targets = targets[sorted_indices]
    pred = pred[sorted_indices]

    plt.plot(times, targets, label='Ground Truth', linestyle='--', color='k')
    plt.plot(times, pred, label='Prediction', linestyle='-', color='r')
    plt.title(f"Prediction at epoch {epoch}")
    plt.legend()
    plt.savefig(f"figures/{path}/training_process/{epoch}.png")
    plt.close()