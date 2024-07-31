import torch

def update_weights(model, path):  
    model_weights = torch.load(f'weights/{path}/weights.pth')

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in model_weights:
                param.copy_(model_weights[name])

    return model