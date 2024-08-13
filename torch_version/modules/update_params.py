import torch

def update_weights(model, path):
    try:
        # Attempt to load weights from the primary path
        model_weights = torch.load(f'weights/{path}/weights.pth')
    except FileNotFoundError:
        # If the file is not found, load weights from the alternative path
        model_weights = torch.load(f'torch_version/weights/{path}/weights.pth') 
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in model_weights:
                param.copy_(model_weights[name])

    return model