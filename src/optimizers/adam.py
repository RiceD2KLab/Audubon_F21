import torch


def get_adam_optim(model, lr=0.0001):
    '''
    Returns an Adam optimizer associated with the input model with learning rate lr

    Args:
        model (nn.Module): PyTorch model to associate the optimizer with
        lr (float): Learning rate to use for the optimizer (default: 0.0001)

    Returns:
        Adam optimizer instance
    '''
    # Filter model parameters that need to have gradients computed
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer
