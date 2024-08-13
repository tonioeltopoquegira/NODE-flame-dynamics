import numpy as np
import matplotlib.pyplot as plt


from modules.utils import mse_loss, data_loader


# Validation file

def validate(run, restored_params, model, path, dataset):
    model_name = run['model_name']
    num_traj = 2
    err = 0.0
    num_obs = 0

    
    if run['model']=='node': 
        for (times, x, y, iv), en in zip(dataset, range(num_traj)):
            pred = model(times, x, iv)
            plt.figure(figsize=(16, 8))
            plt.plot(y, label='Ground Truth', linestyle='-', color='k')
            plt.plot(pred, label='Prediction', linestyle='-', color='r')
            plt.title("Prediction")
            plt.legend()
            plt.savefig(f"figures/{path}/test_results/traj_{en+1}")
            plt.close()
            err += np.sum((pred - y)**2)
            num_obs += len(y)
        
        if run['time-dep'] == 'time-att':
            model.plot_attention(path, 999999)
    
    else:
        data_load = data_loader(*dataset, batch_size=200)
        for en, (t, x,y) in enumerate(data_load):
            sorted_indices = np.argsort(t)
            t = t[sorted_indices]
            y = y[sorted_indices]
            pred = model.apply({'params': restored_params}, x)
            pred = pred[sorted_indices]
            plt.figure(figsize=(16, 8))
            plt.plot(y, label='Ground Truth', linestyle='-', color='k')
            plt.plot(pred, label='Prediction', linestyle='-', color='r')
            plt.title("Prediction")
            plt.legend()
            plt.savefig(f"figures/{path}/test_results/traj_{en+1}")
            if run['show_res']:
                plt.show()
            plt.close()
            err += np.sum((pred - y)**2)
            num_obs += len(y)
        
        if run['time-dep'] == 'time-att':
            model.plot_attention(path, 999999)
        
    print(f"Mean Square Error Test Data: {err/num_obs}")
        

