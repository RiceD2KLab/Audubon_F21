import torch


def get_adam_optim(model, lr=0.0001):
    ''' Return an optimizer associated with the input model '''
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer
